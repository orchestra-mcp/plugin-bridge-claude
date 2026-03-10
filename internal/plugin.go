package internal

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"sync"

	"github.com/orchestra-mcp/plugin-bridge-claude/internal/tools"
	"github.com/orchestra-mcp/sdk-go/plugin"
)

// BridgePlugin manages Claude Code CLI processes and registers all bridge tools.
type BridgePlugin struct {
	active  map[string]*ClaudeProcess
	mu      sync.RWMutex
	permSrv *PermissionServer // kept for backwards compat; will be removed
}

// NewBridgePlugin creates a new BridgePlugin with an empty active process map.
func NewBridgePlugin() *BridgePlugin {
	return &BridgePlugin{
		active:  make(map[string]*ClaudeProcess),
		permSrv: NewPermissionServer(),
	}
}

// StartPermissionServer starts the legacy HTTP permission server.
// Kept for backwards compatibility but no longer used for the main flow.
func (bp *BridgePlugin) StartPermissionServer(ctx context.Context) error {
	return bp.permSrv.Start(ctx)
}

// RegisterTools registers all bridge tools with the plugin builder.
func (bp *BridgePlugin) RegisterTools(builder *plugin.PluginBuilder) {
	bridge := &tools.Bridge{
		Spawn:           bp.spawnAdapter,
		SpawnAsync:      bp.spawnAsyncAdapter,
		SpawnBackground: bp.spawnBackgroundAdapter,
		SpawnStream:     bp.spawnStreamAdapter,
		Plugin:          bp,
	}

	// Permission store — drains from active process channels.
	permStore := tools.NewPermissionStore(bp)

	// --- Prompt tools (2) ---
	builder.RegisterTool("ai_prompt",
		"Send a one-shot prompt to Claude Code CLI and return the response",
		tools.AIPromptSchema(), tools.AIPrompt(bridge))

	builder.RegisterStreamingTool("ai_prompt_stream",
		"Stream a one-shot prompt to Claude Code CLI, yielding text chunks as they arrive",
		tools.AIPromptStreamSchema(), tools.AIPromptStream(bridge))

	builder.RegisterStreamingTool("chat_stream",
		"Stream a chat session with Claude Code CLI, yielding granular ChatEvent JSON chunks (text, tool_start, tool_end, status, result, error)",
		tools.ChatStreamSchema(), tools.ChatStream(bridge))

	// --- Session tools (4) ---
	builder.RegisterTool("spawn_session",
		"Spawn a persistent Claude Code CLI session with a prompt",
		tools.SpawnSessionSchema(), tools.SpawnSession(bridge))

	builder.RegisterTool("kill_session",
		"Kill a running Claude Code CLI session",
		tools.KillSessionSchema(), tools.KillSession(bridge))

	builder.RegisterTool("session_status",
		"Check the status of a Claude Code CLI session",
		tools.SessionStatusSchema(), tools.SessionStatus(bridge))

	builder.RegisterTool("list_active",
		"List all active Claude Code CLI sessions",
		tools.ListActiveSchema(), tools.ListActive(bridge))

	// --- Permission tools (2) — now backed by stdin/stdout control messages ---
	builder.RegisterTool("get_pending_permission",
		"Get pending permission and question requests waiting for user approval",
		tools.GetPendingPermissionSchema(), tools.GetPendingPermission(permStore))

	builder.RegisterTool("respond_permission",
		"Respond to a pending permission or question request with approve/deny/answer",
		tools.RespondPermissionSchema(), tools.RespondPermission(permStore))

	builder.RegisterTool("drain_session_events",
		"Non-blocking drain of all pending chat events from active Claude sessions",
		tools.DrainSessionEventsSchema(), tools.DrainSessionEvents(permStore))
}

// --- PermissionBridge interface implementation ---
// This allows the tools package to drain permission/question requests from
// active ClaudeProcess instances without importing internal types directly.

// DrainPendingPermissions non-blocking drains all pending StdioPermissionRequest
// items from all active processes' PermissionCh channels.
func (bp *BridgePlugin) DrainPendingPermissions() []tools.StdioPermissionRequest {
	bp.mu.RLock()
	defer bp.mu.RUnlock()

	var result []tools.StdioPermissionRequest
	for _, proc := range bp.active {
		if proc.PermissionCh == nil {
			continue
		}
		for {
			select {
			case pr, ok := <-proc.PermissionCh:
				if !ok {
					goto nextProc
				}
				result = append(result, tools.StdioPermissionRequest{
					RequestID: pr.RequestID,
					ToolName:  pr.ToolName,
					ToolInput: pr.ToolInput,
					Reason:    pr.Reason,
					ToolUseID: pr.ToolUseID,
				})
			default:
				goto nextProc
			}
		}
	nextProc:
	}
	return result
}

// DrainPendingQuestions non-blocking drains all pending QuestionRequest items
// from all active processes' QuestionCh channels.
func (bp *BridgePlugin) DrainPendingQuestions() []tools.StdioQuestionRequest {
	bp.mu.RLock()
	defer bp.mu.RUnlock()

	var result []tools.StdioQuestionRequest
	for _, proc := range bp.active {
		if proc.QuestionCh == nil {
			continue
		}
		for {
			select {
			case qr, ok := <-proc.QuestionCh:
				if !ok {
					goto nextQProc
				}
				questionsJSON, _ := json.Marshal(qr.Questions)
				result = append(result, tools.StdioQuestionRequest{
					RequestID: qr.RequestID,
					ToolUseID: qr.ToolUseID,
					Questions: questionsJSON,
					RawInput:  qr.RawInput,
				})
			default:
				goto nextQProc
			}
		}
	nextQProc:
	}
	return result
}

// DrainSessionEvents non-blocking drains all pending ChatEvent items from all
// active processes' EventCh channels.
func (bp *BridgePlugin) DrainSessionEvents() []tools.ChatEvent {
	bp.mu.RLock()
	defer bp.mu.RUnlock()

	var result []tools.ChatEvent
	for _, proc := range bp.active {
		if proc.EventCh == nil {
			continue
		}
		for {
			select {
			case ev, ok := <-proc.EventCh:
				if !ok {
					goto nextEvProc
				}
				result = append(result, tools.ChatEvent{
					Type:       tools.ChatEventType(ev.Type),
					SessionID:  ev.SessionID,
					Text:       ev.Text,
					ToolName:   ev.ToolName,
					ToolID:     ev.ToolID,
					ToolInput:  ev.ToolInput,
					ToolError:  ev.ToolError,
					TokensIn:   ev.TokensIn,
					TokensOut:  ev.TokensOut,
					CostUSD:    ev.CostUSD,
					ModelUsed:  ev.ModelUsed,
					DurationMs: ev.DurationMs,
					RequestID:  ev.RequestID,
					Reason:     ev.Reason,
				})
			default:
				goto nextEvProc
			}
		}
	nextEvProc:
	}
	return result
}

// RespondPermission finds the active process that owns the given requestID and
// sends the permission decision via stdin control_response. toolInput is the
// original tool arguments that must be echoed back so Claude CLI knows what
// to execute.
func (bp *BridgePlugin) RespondPermission(requestID string, approved bool, toolInput json.RawMessage) bool {
	bp.mu.RLock()
	defer bp.mu.RUnlock()

	for _, proc := range bp.active {
		if proc.IsRunning() {
			// Try writing — if this process doesn't own the requestID,
			// the CLI will ignore the response (harmless).
			if err := proc.WritePermission(requestID, approved, toolInput); err == nil {
				return true
			}
		}
	}
	return false
}

// HasRunningProcesses returns true if any Claude process is currently active.
func (bp *BridgePlugin) HasRunningProcesses() bool {
	bp.mu.RLock()
	defer bp.mu.RUnlock()
	for _, proc := range bp.active {
		if proc.IsRunning() {
			return true
		}
	}
	return false
}

// RespondQuestion finds the active process and sends the question answer.
// rawInput is the original AskUserQuestion tool input from the control_request.
func (bp *BridgePlugin) RespondQuestion(requestID, answer string, rawInput json.RawMessage) bool {
	log.Printf("[bridge] RespondQuestion: acquiring RLock for id=%s", requestID[:min(len(requestID), 8)])
	bp.mu.RLock()
	defer bp.mu.RUnlock()
	log.Printf("[bridge] RespondQuestion: RLock acquired, %d active procs", len(bp.active))

	for sid, proc := range bp.active {
		if proc.IsRunning() {
			log.Printf("[bridge] RespondQuestion: writing to proc %s", sid[:min(len(sid), 8)])
			if err := proc.WriteQuestion(requestID, answer, rawInput); err == nil {
				log.Printf("[bridge] RespondQuestion: success")
				return true
			} else {
				log.Printf("[bridge] RespondQuestion: write error: %v", err)
			}
		}
	}
	log.Printf("[bridge] RespondQuestion: no running process accepted the answer")
	return false
}

// --- Spawn adapters ---

func (bp *BridgePlugin) spawnAdapter(ctx context.Context, opts tools.SpawnOptions) (*tools.ChatResponse, error) {
	internalOpts := convertOpts(opts)
	resp, err := Spawn(ctx, internalOpts)
	if err != nil {
		return nil, err
	}
	return convertResp(resp), nil
}

func (bp *BridgePlugin) spawnAsyncAdapter(ctx context.Context, opts tools.SpawnOptions) (tools.ProcessHandle, *tools.ChatResponse, error) {
	internalOpts := convertOpts(opts)
	proc, resp, err := SpawnAsync(ctx, internalOpts)
	if err != nil {
		return nil, nil, err
	}
	return wrapProcess(proc), convertResp(resp), nil
}

func (bp *BridgePlugin) spawnBackgroundAdapter(ctx context.Context, opts tools.SpawnOptions) (tools.ProcessHandle, error) {
	internalOpts := convertOpts(opts)
	proc, err := SpawnBackground(ctx, internalOpts)
	if err != nil {
		return nil, err
	}
	return wrapProcess(proc), nil
}

func (bp *BridgePlugin) spawnStreamAdapter(ctx context.Context, opts tools.SpawnOptions, chunkFn func([]byte)) (*tools.ChatResponse, error) {
	internalOpts := convertOpts(opts)
	resp, err := SpawnStream(ctx, internalOpts, chunkFn)
	if err != nil {
		return nil, err
	}
	return convertResp(resp), nil
}

// --- BridgePluginInterface implementation ---

func (bp *BridgePlugin) TrackProcess(proc tools.ProcessHandle) {
	cp := unwrapProcess(proc)
	if cp == nil {
		return
	}
	bp.mu.Lock()
	defer bp.mu.Unlock()
	bp.active[cp.SessionID] = cp
}

func (bp *BridgePlugin) GetProcess(sessionID string) tools.ProcessHandle {
	bp.mu.RLock()
	defer bp.mu.RUnlock()
	proc, ok := bp.active[sessionID]
	if !ok {
		return nil
	}
	return wrapProcess(proc)
}

func (bp *BridgePlugin) RemoveProcess(sessionID string) tools.ProcessHandle {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	proc, ok := bp.active[sessionID]
	if !ok {
		return nil
	}
	delete(bp.active, sessionID)
	return wrapProcess(proc)
}

func (bp *BridgePlugin) ListProcesses() []tools.ProcessHandle {
	bp.mu.RLock()
	defer bp.mu.RUnlock()
	procs := make([]tools.ProcessHandle, 0, len(bp.active))
	for _, proc := range bp.active {
		procs = append(procs, wrapProcess(proc))
	}
	return procs
}

func (bp *BridgePlugin) KillAll() {
	bp.mu.Lock()
	for id, proc := range bp.active {
		if proc.Cmd != nil && proc.Cmd.Process != nil {
			_ = proc.Cmd.Process.Signal(os.Kill)
		}
		delete(bp.active, id)
	}
	bp.mu.Unlock()

	if bp.permSrv != nil {
		bp.permSrv.Stop()
	}
}

// --- Process adapter ---

type processAdapter struct {
	cp *ClaudeProcess
}

func (a *processAdapter) IsRunning() bool          { return a.cp.IsRunning() }
func (a *processAdapter) GetSessionID() string      { return a.cp.GetSessionID() }
func (a *processAdapter) SetSessionID(id string)    { a.cp.SetSessionID(id) }
func (a *processAdapter) GetPID() int               { return a.cp.GetPID() }
func (a *processAdapter) GetStartedAt() string      { return a.cp.GetStartedAt() }
func (a *processAdapter) GetUptimeSeconds() float64 { return a.cp.GetUptimeSeconds() }
func (a *processAdapter) Kill() error               { return a.cp.Kill() }
func (a *processAdapter) SetAutoApprove(v bool)     { a.cp.SetAutoApprove(v) }

func (a *processAdapter) GetEventCh() <-chan tools.ChatEvent {
	if a.cp.EventCh == nil {
		return nil
	}
	// Bridge internal ChatEvent → tools ChatEvent via a converter goroutine.
	// The internal EventCh is buffered(64), so we mirror that.
	out := make(chan tools.ChatEvent, 64)
	go func() {
		defer close(out)
		for ev := range a.cp.EventCh {
			out <- tools.ChatEvent{
				Type:       tools.ChatEventType(ev.Type),
				SessionID:  ev.SessionID,
				Text:       ev.Text,
				ToolName:   ev.ToolName,
				ToolID:     ev.ToolID,
				ToolInput:  ev.ToolInput,
				ToolError:  ev.ToolError,
				TokensIn:   ev.TokensIn,
				TokensOut:  ev.TokensOut,
				CostUSD:    ev.CostUSD,
				ModelUsed:  ev.ModelUsed,
				DurationMs: ev.DurationMs,
				RequestID:  ev.RequestID,
				Reason:     ev.Reason,
			}
		}
	}()
	return out
}

func (a *processAdapter) GetResponse() *tools.ChatResponse {
	return convertResp(a.cp.GetResponse())
}

func (a *processAdapter) WaitResponse(ctx context.Context) (*tools.ChatResponse, error) {
	resp, err := a.cp.WaitResponse(ctx)
	if err != nil {
		return nil, err
	}
	return convertResp(resp), nil
}

func wrapProcess(cp *ClaudeProcess) tools.ProcessHandle {
	if cp == nil {
		return nil
	}
	return &processAdapter{cp: cp}
}

func unwrapProcess(proc tools.ProcessHandle) *ClaudeProcess {
	if a, ok := proc.(*processAdapter); ok {
		return a.cp
	}
	return nil
}

// --- Type conversion helpers ---

func convertOpts(opts tools.SpawnOptions) SpawnOptions {
	return SpawnOptions{
		SessionID:      opts.SessionID,
		Resume:         opts.Resume,
		Prompt:         opts.Prompt,
		Model:          opts.Model,
		Workspace:      opts.Workspace,
		AllowedTools:   opts.AllowedTools,
		PermissionMode: opts.PermissionMode,
		MaxBudget:      opts.MaxBudget,
		SystemPrompt:   opts.SystemPrompt,
		Env:            opts.Env,
	}
}

func convertResp(resp *ChatResponse) *tools.ChatResponse {
	if resp == nil {
		return nil
	}
	return &tools.ChatResponse{
		ResponseText: resp.ResponseText,
		TokensIn:     resp.TokensIn,
		TokensOut:    resp.TokensOut,
		CostUSD:      resp.CostUSD,
		ModelUsed:    resp.ModelUsed,
		DurationMs:   resp.DurationMs,
		SessionID:    resp.SessionID,
	}
}
