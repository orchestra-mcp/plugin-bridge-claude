package internal

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// PermissionRequest holds the data for a pending tool-use permission request
// from the Claude Code PreToolUse hook.
type PermissionRequest struct {
	ID        string         `json:"id"`
	ToolName  string         `json:"tool_name"`
	ToolInput map[string]any `json:"tool_input"`
	SessionID string         `json:"session_id"`
	Cwd       string         `json:"cwd"`
	CreatedAt time.Time      `json:"created_at"`
	decideCh  chan string     // "approve" or "deny"
}

// PermissionServer is an HTTP server that receives PreToolUse hook events from
// the Claude Code CLI and holds them until the user responds via the Swift UI.
// It binds to a random localhost port and writes the port to a well-known file
// so the hook script can discover it.
type PermissionServer struct {
	mu       sync.Mutex
	pending  map[string]*PermissionRequest // id → request
	server   *http.Server
	listener net.Listener
	portFile string
}

// portFilePath returns the path to the port file used by the hook script.
func portFilePath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".orchestra", "permission-server.port")
}

// NewPermissionServer creates a new permission server. Call Start() to begin
// listening.
func NewPermissionServer() *PermissionServer {
	ps := &PermissionServer{
		pending:  make(map[string]*PermissionRequest),
		portFile: portFilePath(),
	}
	return ps
}

// Start begins listening on a random localhost port and writes the port to the
// port file so the hook script can find it. Call Stop() to shut down.
func (ps *PermissionServer) Start(ctx context.Context) error {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("permission server: listen: %w", err)
	}
	ps.listener = ln

	port := ln.Addr().(*net.TCPAddr).Port

	// Write the port to the well-known file.
	if err := os.MkdirAll(filepath.Dir(ps.portFile), 0755); err != nil {
		ln.Close()
		return fmt.Errorf("permission server: mkdir: %w", err)
	}
	if err := os.WriteFile(ps.portFile, []byte(fmt.Sprintf("%d", port)), 0644); err != nil {
		ln.Close()
		return fmt.Errorf("permission server: write port file: %w", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/permission", ps.handlePermission)
	mux.HandleFunc("/permission/respond", ps.handleRespond)
	mux.HandleFunc("/permission/pending", ps.handlePending)

	ps.server = &http.Server{Handler: mux}

	go func() {
		if err := ps.server.Serve(ln); err != nil && err != http.ErrServerClosed {
			log.Printf("[permission] server error: %v", err)
		}
	}()

	go func() {
		<-ctx.Done()
		ps.Stop()
	}()

	log.Printf("[permission] server listening on port %d", port)
	return nil
}

// Stop shuts down the permission server and removes the port file. Any pending
// permission requests are auto-approved so the Claude processes are not stuck.
func (ps *PermissionServer) Stop() {
	if ps.server != nil {
		_ = ps.server.Shutdown(context.Background())
	}
	_ = os.Remove(ps.portFile)

	// Auto-approve any pending requests so hook scripts don't hang.
	ps.mu.Lock()
	defer ps.mu.Unlock()
	for _, req := range ps.pending {
		select {
		case req.decideCh <- "approve":
		default:
		}
	}
	ps.pending = make(map[string]*PermissionRequest)
}

// handlePermission is called by the PreToolUse hook script. It registers a new
// pending permission request and blocks until the user responds (or 5 min timeout).
func (ps *PermissionServer) handlePermission(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var body struct {
		ToolName  string         `json:"tool_name"`
		ToolInput map[string]any `json:"tool_input"`
		SessionID string         `json:"session_id"`
		Cwd       string         `json:"cwd"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}

	req := &PermissionRequest{
		ID:        fmt.Sprintf("perm-%d", time.Now().UnixNano()),
		ToolName:  body.ToolName,
		ToolInput: body.ToolInput,
		SessionID: body.SessionID,
		Cwd:       body.Cwd,
		CreatedAt: time.Now(),
		decideCh:  make(chan string, 1),
	}

	ps.mu.Lock()
	ps.pending[req.ID] = req
	ps.mu.Unlock()

	log.Printf("[permission] pending %s: tool=%s session=%s", req.ID, req.ToolName, req.SessionID)

	// Block until the user responds or the request context times out (5 min).
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	var decision string
	select {
	case decision = <-req.decideCh:
	case <-ctx.Done():
		decision = "approve" // timeout → allow
	}

	ps.mu.Lock()
	delete(ps.pending, req.ID)
	ps.mu.Unlock()

	log.Printf("[permission] %s: decision=%s", req.ID, decision)

	w.Header().Set("Content-Type", "application/json")
	if decision == "deny" {
		_ = json.NewEncoder(w).Encode(map[string]string{"decision": "deny", "reason": "Permission denied by user"})
	} else {
		_ = json.NewEncoder(w).Encode(map[string]string{"decision": "approve"})
	}
}

// handlePending returns the list of pending permission requests (polled by the Swift UI).
func (ps *PermissionServer) handlePending(w http.ResponseWriter, r *http.Request) {
	ps.mu.Lock()
	reqs := make([]*PermissionRequest, 0, len(ps.pending))
	for _, req := range ps.pending {
		reqs = append(reqs, req)
	}
	ps.mu.Unlock()

	type out struct {
		ID        string         `json:"id"`
		ToolName  string         `json:"tool_name"`
		ToolInput map[string]any `json:"tool_input"`
		SessionID string         `json:"session_id"`
		Cwd       string         `json:"cwd"`
		CreatedAt string         `json:"created_at"`
	}
	var result []out
	for _, req := range reqs {
		result = append(result, out{
			ID:        req.ID,
			ToolName:  req.ToolName,
			ToolInput: req.ToolInput,
			SessionID: req.SessionID,
			Cwd:       req.Cwd,
			CreatedAt: req.CreatedAt.Format(time.RFC3339),
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(result)
}

// handleRespond receives a user decision for a pending permission request.
func (ps *PermissionServer) handleRespond(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var body struct {
		ID       string `json:"id"`
		Decision string `json:"decision"` // "approve" or "deny"
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}

	ps.mu.Lock()
	req, ok := ps.pending[body.ID]
	ps.mu.Unlock()

	if !ok {
		http.Error(w, "permission request not found", http.StatusNotFound)
		return
	}

	decision := body.Decision
	if decision != "approve" && decision != "deny" {
		decision = "approve"
	}

	select {
	case req.decideCh <- decision:
	default:
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// Port returns the port the server is listening on, or 0 if not started.
func (ps *PermissionServer) Port() int {
	if ps.listener == nil {
		return 0
	}
	return ps.listener.Addr().(*net.TCPAddr).Port
}

// GetPending returns the current pending permission requests as JSON-serializable data.
func (ps *PermissionServer) GetPending() []map[string]any {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	var result []map[string]any
	for _, req := range ps.pending {
		result = append(result, map[string]any{
			"id":         req.ID,
			"tool_name":  req.ToolName,
			"tool_input": req.ToolInput,
			"session_id": req.SessionID,
			"cwd":        req.Cwd,
			"created_at": req.CreatedAt.Format(time.RFC3339),
		})
	}
	return result
}

// Respond sends a decision for a pending permission request. Returns false if
// the request ID is not found.
func (ps *PermissionServer) Respond(id, decision string) bool {
	ps.mu.Lock()
	req, ok := ps.pending[id]
	ps.mu.Unlock()

	if !ok {
		return false
	}

	if decision != "approve" && decision != "deny" {
		decision = "approve"
	}

	select {
	case req.decideCh <- decision:
		return true
	default:
		return false
	}
}
