// Package internal contains the core logic for the bridge.claude plugin.
// It spawns Claude Code CLI processes in interactive stream-json mode, parses
// their output, handles permission requests via stdin/stdout control messages,
// and manages active session lifecycles.
//
// Architecture (matching orch-ref/app/ai/bridge.go):
//
//	Claude CLI runs with --input-format stream-json --output-format stream-json
//	--permission-prompt-tool stdio. Permission requests and AskUserQuestion
//	prompts arrive as control_request events on stdout. We respond via stdin
//	with control_response messages. No PreToolUse hooks needed.
package internal

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// SpawnOptions configures how a Claude Code CLI process is launched.
type SpawnOptions struct {
	SessionID      string            // --session-id (empty for one-shot ai_prompt)
	Resume         bool              // use --resume instead of --session-id
	Prompt         string            // the user message
	Model          string            // --model (optional)
	Workspace      string            // working directory for claude process
	AllowedTools   []string          // --allowedTools (optional)
	PermissionMode string            // --permission-mode (optional, ignored in interactive mode)
	MaxBudget      float64           // --max-budget-usd (optional)
	SystemPrompt   string            // --system-prompt (optional)
	Env            map[string]string // extra env vars (ANTHROPIC_API_KEY, etc.)
}

// ChatResponse holds the result of a completed Claude Code CLI invocation.
type ChatResponse struct {
	ResponseText string  `json:"response_text"`
	TokensIn     int64   `json:"tokens_in"`
	TokensOut    int64   `json:"tokens_out"`
	CostUSD      float64 `json:"cost_usd"`
	ModelUsed    string  `json:"model_used"`
	DurationMs   int64   `json:"duration_ms"`
	SessionID    string  `json:"session_id"`
}

// StdioPermissionRequest is emitted when Claude needs permission for a tool call
// via the interactive stream-json protocol (control_request events on stdout).
// This is distinct from PermissionRequest in permission.go (hook-based system).
type StdioPermissionRequest struct {
	RequestID string          `json:"requestId"`
	ToolName  string          `json:"toolName"`
	ToolInput json.RawMessage `json:"toolInput"`
	Reason    string          `json:"reason"`
	ToolUseID string          `json:"toolUseId"`
}

// QuestionRequest is emitted when Claude uses AskUserQuestion.
type QuestionRequest struct {
	RequestID string         `json:"requestId"`
	ToolUseID string         `json:"toolUseId"`
	Questions []QuestionItem `json:"questions"`
	RawInput  json.RawMessage `json:"rawInput,omitempty"`
}

// QuestionItem is a single question in an AskUserQuestion prompt.
type QuestionItem struct {
	Question    string           `json:"question"`
	Options     []QuestionOption `json:"options"`
	Header      string           `json:"header,omitempty"`
	MultiSelect bool             `json:"multiSelect,omitempty"`
}

// QuestionOption is a single option in a question.
type QuestionOption struct {
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
}

// ChatEventType classifies streaming events for the Swift UI.
type ChatEventType string

const (
	EventTextChunk  ChatEventType = "text_chunk"
	EventToolStart  ChatEventType = "tool_start"
	EventToolEnd    ChatEventType = "tool_end"
	EventThinking   ChatEventType = "thinking"
	EventStatus     ChatEventType = "status"
	EventResult     ChatEventType = "result"
	EventError      ChatEventType = "error"
	EventPermission ChatEventType = "permission"
	EventQuestion   ChatEventType = "question"
)

// ChatEvent is a single granular event emitted during a Claude session.
type ChatEvent struct {
	Type       ChatEventType `json:"type"`
	SessionID  string        `json:"session_id"`
	Text       string        `json:"text,omitempty"`
	ToolName   string        `json:"tool_name,omitempty"`
	ToolID     string        `json:"tool_id,omitempty"`
	ToolInput  string        `json:"tool_input,omitempty"`
	ToolError  bool          `json:"tool_error,omitempty"`
	TokensIn   int64         `json:"tokens_in,omitempty"`
	TokensOut  int64         `json:"tokens_out,omitempty"`
	CostUSD    float64       `json:"cost_usd,omitempty"`
	ModelUsed  string        `json:"model_used,omitempty"`
	DurationMs int64         `json:"duration_ms,omitempty"`
	// Permission/Question fields
	RequestID string `json:"request_id,omitempty"`
	Reason    string `json:"reason,omitempty"`
}

// safeTools are auto-approved without prompting the user.
var safeTools = map[string]bool{
	"Read":          true,
	"Glob":          true,
	"Grep":          true,
	"WebFetch":      true,
	"WebSearch":     true,
	"TodoWrite":     true,
	"EnterPlanMode": true,
	"ExitPlanMode":  true,
	"Task":          true,
	"Skill":         true,
	"NotebookEdit":  true,
}

// ClaudeProcess represents a running Claude Code CLI process.
type ClaudeProcess struct {
	SessionID string
	Cmd       *exec.Cmd
	StartedAt time.Time
	Done      chan struct{}

	// PermissionCh receives permission requests from the scanner goroutine.
	// The Swift UI polls these and responds via WritePermission().
	PermissionCh chan StdioPermissionRequest

	// QuestionCh receives AskUserQuestion requests from the scanner goroutine.
	QuestionCh chan QuestionRequest

	// AutoApprove, when true, causes handleControlRequest to auto-approve
	// all permission requests instead of sending to PermissionCh. This is
	// set for synchronous callers (wait=true) where nobody drains the channel.
	AutoApprove bool

	// EventCh receives granular streaming events. Buffered to 64 so the
	// scanner goroutine doesn't block. TCP streaming handler drains this.
	EventCh chan ChatEvent

	mu       sync.Mutex
	exitErr  error
	finished bool
	response *ChatResponse

	stdinPipe io.WriteCloser
	stdinMu   sync.Mutex
}

// IsRunning reports whether the process is still executing.
func (cp *ClaudeProcess) IsRunning() bool {
	cp.mu.Lock()
	defer cp.mu.Unlock()
	return !cp.finished
}

// GetSessionID returns the session ID of this process.
func (cp *ClaudeProcess) GetSessionID() string {
	return cp.SessionID
}

// GetPID returns the OS process ID, or 0 if the process has not started.
func (cp *ClaudeProcess) GetPID() int {
	if cp.Cmd != nil && cp.Cmd.Process != nil {
		return cp.Cmd.Process.Pid
	}
	return 0
}

// GetStartedAt returns the process start time as an ISO 8601 string.
func (cp *ClaudeProcess) GetStartedAt() string {
	return cp.StartedAt.Format(time.RFC3339)
}

// GetUptimeSeconds returns the number of seconds since the process started.
func (cp *ClaudeProcess) GetUptimeSeconds() float64 {
	return time.Since(cp.StartedAt).Seconds()
}

// SetAutoApprove enables or disables auto-approval of all permission requests.
// Synchronous callers (wait=true) must enable this because nobody drains
// PermissionCh, which would otherwise deadlock.
func (cp *ClaudeProcess) SetAutoApprove(v bool) {
	cp.AutoApprove = v
}

// Kill sends a kill signal to the underlying OS process.
func (cp *ClaudeProcess) Kill() error {
	if cp.Cmd != nil && cp.Cmd.Process != nil {
		return cp.Cmd.Process.Kill()
	}
	return fmt.Errorf("process not running")
}

// markDone records the exit error and response, then signals completion.
func (cp *ClaudeProcess) markDone(err error, resp *ChatResponse) {
	cp.mu.Lock()
	defer cp.mu.Unlock()
	cp.exitErr = err
	cp.response = resp
	cp.finished = true
	close(cp.Done)
	// Close channels so consumers unblock.
	if cp.PermissionCh != nil {
		close(cp.PermissionCh)
		cp.PermissionCh = nil
	}
	if cp.QuestionCh != nil {
		close(cp.QuestionCh)
		cp.QuestionCh = nil
	}
	if cp.EventCh != nil {
		close(cp.EventCh)
		cp.EventCh = nil
	}
}

// GetResponse returns the response if the process has finished, nil otherwise.
func (cp *ClaudeProcess) GetResponse() *ChatResponse {
	cp.mu.Lock()
	defer cp.mu.Unlock()
	if !cp.finished {
		return nil
	}
	return cp.response
}

// WaitResponse blocks until the process completes and returns the response.
func (cp *ClaudeProcess) WaitResponse(ctx context.Context) (*ChatResponse, error) {
	select {
	case <-cp.Done:
		cp.mu.Lock()
		defer cp.mu.Unlock()
		return cp.response, cp.exitErr
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// WritePermission sends a permission decision back to the Claude process via
// stdin. This unblocks the CLI which is waiting for a control_response.
func (cp *ClaudeProcess) WritePermission(requestID string, approved bool, toolInput json.RawMessage) error {
	var controlResp map[string]any
	if approved {
		updatedInput := map[string]any{}
		if len(toolInput) > 0 {
			_ = json.Unmarshal(toolInput, &updatedInput)
		}
		controlResp = map[string]any{
			"type": "control_response",
			"response": map[string]any{
				"subtype":    "success",
				"request_id": requestID,
				"response": map[string]any{
					"behavior":     "allow",
					"updatedInput": updatedInput,
				},
			},
		}
	} else {
		controlResp = map[string]any{
			"type": "control_response",
			"response": map[string]any{
				"subtype":    "success",
				"request_id": requestID,
				"response": map[string]any{
					"behavior": "deny",
					"message":  "User denied from desktop",
				},
			},
		}
	}
	return cp.writeStdinJSON(controlResp)
}

// WriteQuestion sends an AskUserQuestion answer back to the Claude process.
func (cp *ClaudeProcess) WriteQuestion(requestID, answer string) error {
	controlResp := map[string]any{
		"type": "control_response",
		"response": map[string]any{
			"subtype":    "success",
			"request_id": requestID,
			"response": map[string]any{
				"answer": answer,
			},
		},
	}
	return cp.writeStdinJSON(controlResp)
}

// writeStdinJSON marshals data as JSON and writes it to the Claude process stdin.
func (cp *ClaudeProcess) writeStdinJSON(data any) error {
	cp.stdinMu.Lock()
	defer cp.stdinMu.Unlock()
	if cp.stdinPipe == nil {
		return fmt.Errorf("stdin pipe not available")
	}
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	_, err = cp.stdinPipe.Write(b)
	return err
}

// writeUserMessage sends the initial user message via stdin (stream-json format).
func (cp *ClaudeProcess) writeUserMessage(prompt string) error {
	msg := map[string]any{
		"type": "user",
		"message": map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": prompt},
			},
		},
	}
	return cp.writeStdinJSON(msg)
}

// closeStdin closes the stdin pipe.
func (cp *ClaudeProcess) closeStdin() {
	cp.stdinMu.Lock()
	defer cp.stdinMu.Unlock()
	if cp.stdinPipe != nil {
		_ = cp.stdinPipe.Close()
		cp.stdinPipe = nil
	}
}

// controlRequestBody is the parsed body of a control_request event.
type controlRequestBody struct {
	Subtype   string          `json:"subtype"`
	ToolName  string          `json:"tool_name"`
	Input     json.RawMessage `json:"input"`
	Reason    string          `json:"decision_reason"`
	ToolUseID string          `json:"tool_use_id"`
}

// askUserQuestionInput is the input schema for AskUserQuestion.
type askUserQuestionInput struct {
	Questions []struct {
		Question    string           `json:"question"`
		Options     []QuestionOption `json:"options"`
		Header      string           `json:"header,omitempty"`
		MultiSelect bool             `json:"multiSelect,omitempty"`
	} `json:"questions"`
}

// Spawn launches a Claude Code CLI process in interactive stream-json mode,
// sends the initial user message via stdin, reads the stream-json output, and
// returns a ChatResponse when the process exits. Permission requests for safe
// tools are auto-approved; dangerous tools are sent to PermissionCh (but since
// this is synchronous, they are also auto-approved to avoid deadlock).
func Spawn(ctx context.Context, opts SpawnOptions) (*ChatResponse, error) {
	args := buildArgs(opts)

	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Env = buildEnv(opts.Env)

	if opts.Workspace != "" {
		cmd.Dir = opts.Workspace
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("bridge.claude: stdout pipe: %w", err)
	}

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("bridge.claude: stdin pipe: %w", err)
	}

	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	start := time.Now()

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("bridge.claude: start claude: %w", err)
	}

	// Write initial user message via stdin.
	cp := &ClaudeProcess{stdinPipe: stdinPipe}
	if err := cp.writeUserMessage(opts.Prompt); err != nil {
		_ = cmd.Process.Kill()
		return nil, fmt.Errorf("bridge.claude: write user message: %w", err)
	}

	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var (
		responseText strings.Builder
		tokensIn     int64
		tokensOut    int64
		costUSD      float64
		modelUsed    string
		sessionID    = opts.SessionID
		toolIDMap    = make(map[string]string)
	)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || line[0] != '{' {
			continue
		}

		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}

		eventType, _ := event["type"].(string)

		// Handle control_request: auto-approve all in synchronous mode
		// since there's no one to poll PermissionCh.
		if eventType == "control_request" {
			reqID, _ := event["request_id"].(string)
			if reqID != "" {
				// Auto-approve everything in synchronous Spawn.
				_ = cp.WritePermission(reqID, true, nil)
			}
			continue
		}

		parseStreamEvent(event, &responseText, &tokensIn, &tokensOut, &costUSD, &modelUsed, &sessionID, toolIDMap)
	}

	cp.closeStdin()
	waitErr := cmd.Wait()
	durationMs := time.Since(start).Milliseconds()

	if ctx.Err() != nil {
		return nil, fmt.Errorf("bridge.claude: process cancelled: %w", ctx.Err())
	}

	if waitErr != nil && responseText.Len() == 0 {
		stderr := strings.TrimSpace(stderrBuf.String())
		if stderr != "" {
			return nil, fmt.Errorf("bridge.claude: %s", stderr)
		}
		return nil, fmt.Errorf("bridge.claude: process exited with error: %w", waitErr)
	}

	return &ChatResponse{
		ResponseText: strings.TrimSpace(responseText.String()),
		TokensIn:     tokensIn,
		TokensOut:    tokensOut,
		CostUSD:      costUSD,
		ModelUsed:    modelUsed,
		DurationMs:   durationMs,
		SessionID:    sessionID,
	}, nil
}

// SpawnAsync launches a Claude Code CLI process in interactive stream-json mode
// and reads output synchronously. Permission requests for safe tools are
// auto-approved; dangerous tool permissions are sent to PermissionCh for the
// Swift UI to handle. The caller receives the ClaudeProcess handle for tracking.
func SpawnAsync(ctx context.Context, opts SpawnOptions) (*ClaudeProcess, *ChatResponse, error) {
	args := buildArgs(opts)

	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Env = buildEnv(opts.Env)

	if opts.Workspace != "" {
		cmd.Dir = opts.Workspace
	}

	cp := &ClaudeProcess{
		SessionID:    opts.SessionID,
		Cmd:          cmd,
		StartedAt:    time.Now(),
		Done:         make(chan struct{}),
		PermissionCh: make(chan StdioPermissionRequest, 16),
		QuestionCh:   make(chan QuestionRequest, 4),
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("bridge.claude: stdout pipe: %w", err)
	}

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("bridge.claude: stdin pipe: %w", err)
	}
	cp.stdinPipe = stdinPipe

	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	if err := cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("bridge.claude: start claude: %w", err)
	}

	// Write initial user message via stdin.
	if err := cp.writeUserMessage(opts.Prompt); err != nil {
		_ = cmd.Process.Kill()
		return nil, nil, fmt.Errorf("bridge.claude: write user message: %w", err)
	}

	// Read output synchronously, handling control_request events.
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var (
		responseText strings.Builder
		tokensIn     int64
		tokensOut    int64
		costUSD      float64
		modelUsed    string
		sessionID    = opts.SessionID
		toolIDMap    = make(map[string]string)
	)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || line[0] != '{' {
			continue
		}

		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}

		eventType, _ := event["type"].(string)

		if eventType == "control_request" {
			handleControlRequest(cp, event)
			continue
		}

		parseStreamEvent(event, &responseText, &tokensIn, &tokensOut, &costUSD, &modelUsed, &sessionID, toolIDMap)
	}

	cp.closeStdin()
	waitErr := cmd.Wait()
	durationMs := time.Since(cp.StartedAt).Milliseconds()

	text := strings.TrimSpace(responseText.String())
	if text == "" {
		if se := strings.TrimSpace(stderrBuf.String()); se != "" {
			text = "[claude stderr]: " + se
		} else if waitErr != nil {
			text = "[claude exited with error]: " + waitErr.Error()
		}
	}

	resp := &ChatResponse{
		ResponseText: text,
		TokensIn:     tokensIn,
		TokensOut:    tokensOut,
		CostUSD:      costUSD,
		ModelUsed:    modelUsed,
		DurationMs:   durationMs,
		SessionID:    sessionID,
	}

	cp.markDone(waitErr, resp)
	return cp, resp, nil
}

// SpawnBackground launches a Claude Code CLI process in interactive stream-json
// mode and returns immediately. Output is read in a background goroutine.
// Permission requests go to cp.PermissionCh for the Swift UI to handle.
func SpawnBackground(ctx context.Context, opts SpawnOptions) (*ClaudeProcess, error) {
	args := buildArgs(opts)

	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Env = buildEnv(opts.Env)
	if opts.Workspace != "" {
		cmd.Dir = opts.Workspace
	}

	cp := &ClaudeProcess{
		SessionID:    opts.SessionID,
		Cmd:          cmd,
		StartedAt:    time.Now(),
		Done:         make(chan struct{}),
		PermissionCh: make(chan StdioPermissionRequest, 16),
		QuestionCh:   make(chan QuestionRequest, 4),
		EventCh:      make(chan ChatEvent, 64),
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("bridge.claude: stdout pipe: %w", err)
	}

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("bridge.claude: stdin pipe: %w", err)
	}
	cp.stdinPipe = stdinPipe

	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("bridge.claude: start claude: %w", err)
	}

	// Write initial user message via stdin.
	if err := cp.writeUserMessage(opts.Prompt); err != nil {
		_ = cmd.Process.Kill()
		return nil, fmt.Errorf("bridge.claude: write user message: %w", err)
	}

	go func() {
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

		var (
			responseText strings.Builder
			tokensIn     int64
			tokensOut    int64
			costUSD      float64
			modelUsed    string
			sessionID    = opts.SessionID
			toolIDMap    = make(map[string]string)
		)

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || line[0] != '{' {
				continue
			}
			var event map[string]any
			if err := json.Unmarshal([]byte(line), &event); err != nil {
				continue
			}

			eventType, _ := event["type"].(string)

			if eventType == "control_request" {
				handleControlRequest(cp, event)
				continue
			}

			parseStreamEvent(event, &responseText, &tokensIn, &tokensOut, &costUSD, &modelUsed, &sessionID, toolIDMap)
			emitEvent(cp, event, sessionID, toolIDMap)

			// When we receive a "result" event, the turn is complete.
			// Close stdin to signal the interactive process to exit,
			// otherwise cmd.Wait() blocks forever.
			if eventType == "result" {
				cp.closeStdin()
			}
		}

		cp.closeStdin() // ensure closed even if no result event
		waitErr := cmd.Wait()
		durationMs := time.Since(cp.StartedAt).Milliseconds()

		text := strings.TrimSpace(responseText.String())

		// If claude exited with no response, surface the error in the response
		// text so the caller sees what went wrong instead of an empty reply.
		if text == "" {
			stderr := strings.TrimSpace(stderrBuf.String())
			if stderr != "" {
				text = "[bridge.claude error]: " + stderr
			} else if waitErr != nil {
				text = "[bridge.claude error]: process exited with: " + waitErr.Error()
			} else if durationMs < 2000 {
				text = fmt.Sprintf("[bridge.claude error]: process exited in %dms with no output (args: %v)", durationMs, args)
			}
		}

		resp := &ChatResponse{
			ResponseText: text,
			TokensIn:     tokensIn,
			TokensOut:    tokensOut,
			CostUSD:      costUSD,
			ModelUsed:    modelUsed,
			DurationMs:   durationMs,
			SessionID:    sessionID,
		}

		cp.markDone(waitErr, resp)
	}()

	return cp, nil
}

// SpawnStream launches a Claude Code CLI process and calls chunkFn for each
// text chunk as the response streams. Permission requests are auto-approved
// in streaming mode since there's no UI to prompt.
func SpawnStream(ctx context.Context, opts SpawnOptions, chunkFn func([]byte)) (*ChatResponse, error) {
	args := buildArgs(opts)
	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Env = buildEnv(opts.Env)
	if opts.Workspace != "" {
		cmd.Dir = opts.Workspace
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("bridge.claude: stdout pipe: %w", err)
	}

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("bridge.claude: stdin pipe: %w", err)
	}

	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	start := time.Now()

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("bridge.claude: start claude: %w", err)
	}

	// Write initial user message via stdin.
	cp := &ClaudeProcess{stdinPipe: stdinPipe}
	if err := cp.writeUserMessage(opts.Prompt); err != nil {
		_ = cmd.Process.Kill()
		return nil, fmt.Errorf("bridge.claude: write user message: %w", err)
	}

	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var (
		responseText strings.Builder
		tokensIn     int64
		tokensOut    int64
		costUSD      float64
		modelUsed    string
		sessionID    = opts.SessionID
		toolIDMap    = make(map[string]string)
	)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || line[0] != '{' {
			continue
		}

		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}

		eventType, _ := event["type"].(string)

		// Auto-approve all permissions in streaming mode.
		if eventType == "control_request" {
			reqID, _ := event["request_id"].(string)
			if reqID != "" {
				_ = cp.WritePermission(reqID, true, nil)
			}
			continue
		}

		if chunk := extractChunkText(event); chunk != "" && chunkFn != nil {
			chunkFn([]byte(chunk))
		}

		parseStreamEvent(event, &responseText, &tokensIn, &tokensOut, &costUSD, &modelUsed, &sessionID, toolIDMap)
	}

	cp.closeStdin()
	waitErr := cmd.Wait()
	durationMs := time.Since(start).Milliseconds()

	if ctx.Err() != nil {
		return nil, fmt.Errorf("bridge.claude: process cancelled: %w", ctx.Err())
	}
	if waitErr != nil && responseText.Len() == 0 {
		stderr := strings.TrimSpace(stderrBuf.String())
		if stderr != "" {
			return nil, fmt.Errorf("bridge.claude: %s", stderr)
		}
		return nil, fmt.Errorf("bridge.claude: process exited with error: %w", waitErr)
	}

	return &ChatResponse{
		ResponseText: strings.TrimSpace(responseText.String()),
		TokensIn:     tokensIn,
		TokensOut:    tokensOut,
		CostUSD:      costUSD,
		ModelUsed:    modelUsed,
		DurationMs:   durationMs,
		SessionID:    sessionID,
	}, nil
}

// handleControlRequest processes a control_request event from the Claude CLI.
// Safe tools are auto-approved immediately. AskUserQuestion is sent to
// QuestionCh. Dangerous tools are sent to PermissionCh for the Swift UI.
func handleControlRequest(cp *ClaudeProcess, event map[string]any) {
	reqID, _ := event["request_id"].(string)
	if reqID == "" {
		return
	}

	rawRequest, ok := event["request"]
	if !ok {
		// No request body — auto-approve.
		_ = cp.WritePermission(reqID, true, nil)
		return
	}

	// Re-marshal the request field to parse it as controlRequestBody.
	reqBytes, err := json.Marshal(rawRequest)
	if err != nil {
		_ = cp.WritePermission(reqID, true, nil)
		return
	}

	var body controlRequestBody
	if err := json.Unmarshal(reqBytes, &body); err != nil {
		_ = cp.WritePermission(reqID, true, nil)
		return
	}

	// AskUserQuestion: send to QuestionCh so the Swift UI can display the
	// question and send back the user's answer.
	if body.ToolName == "AskUserQuestion" && len(body.Input) > 0 {
		var inp askUserQuestionInput
		if err := json.Unmarshal(body.Input, &inp); err == nil && len(inp.Questions) > 0 {
			items := make([]QuestionItem, len(inp.Questions))
			for i, q := range inp.Questions {
				opts := make([]QuestionOption, len(q.Options))
				for j, o := range q.Options {
					opts[j] = QuestionOption{Label: o.Label, Description: o.Description}
				}
				items[i] = QuestionItem{
					Question:    q.Question,
					Options:     opts,
					Header:      q.Header,
					MultiSelect: q.MultiSelect,
				}
			}
			qr := QuestionRequest{
				RequestID: reqID,
				ToolUseID: body.ToolUseID,
				Questions: items,
				RawInput:  body.Input,
			}
			// Emit question event to EventCh for streaming consumers.
			if cp.EventCh != nil {
				questionText := ""
				if len(inp.Questions) > 0 {
					questionText = inp.Questions[0].Question
				}
				trySendEvent(cp, ChatEvent{
					Type:      "question",
					SessionID: cp.SessionID,
					Text:      questionText,
					RequestID: reqID,
					ToolInput: string(body.Input),
					ToolID:    body.ToolUseID,
				})
			}
			// Try to send to QuestionCh; if buffer is full, auto-approve to
			// avoid blocking the scanner goroutine.
			select {
			case cp.QuestionCh <- qr:
				return // Swift UI will call WriteQuestion() later
			default:
				// Buffer full — auto-approve with first option.
				_ = cp.WritePermission(reqID, true, nil)
				return
			}
		}
	}

	// Safe tools: auto-approve.
	if safeTools[body.ToolName] {
		_ = cp.WritePermission(reqID, true, nil)
		return
	}

	// Auto-approve mode: synchronous callers (wait=true) have nobody
	// draining PermissionCh, so approve immediately to avoid deadlock.
	if cp.AutoApprove {
		_ = cp.WritePermission(reqID, true, nil)
		return
	}

	// Dangerous tools (Bash, Write, Edit, etc.): send to PermissionCh.
	pr := StdioPermissionRequest{
		RequestID: reqID,
		ToolName:  body.ToolName,
		ToolInput: body.Input,
		Reason:    body.Reason,
		ToolUseID: body.ToolUseID,
	}

	// Also emit a permission event to EventCh so streaming consumers
	// can present the permission UI without polling PermissionCh.
	if cp.EventCh != nil {
		inputStr := ""
		if len(body.Input) > 0 {
			inputStr = string(body.Input)
		}
		trySendEvent(cp, ChatEvent{
			Type:      "permission",
			SessionID: cp.SessionID,
			ToolName:  body.ToolName,
			ToolInput: inputStr,
			RequestID: reqID,
			Reason:    body.Reason,
			ToolID:    body.ToolUseID,
		})
	}

	// Try to send to PermissionCh; if buffer is full, auto-approve.
	select {
	case cp.PermissionCh <- pr:
		// Swift UI will call WritePermission() later
	default:
		// Buffer full — auto-approve to avoid blocking.
		_ = cp.WritePermission(reqID, true, nil)
	}
}

// emitEvent maps a raw Claude stream-json event to a ChatEvent and sends it
// to EventCh. Dropped silently if the buffer is full.
func emitEvent(cp *ClaudeProcess, event map[string]any, sessionID string, toolIDMap map[string]string) {
	eventType, _ := event["type"].(string)

	switch eventType {
	case "content_block_delta":
		if delta, ok := event["delta"].(map[string]any); ok {
			if text, ok := delta["text"].(string); ok && text != "" {
				trySendEvent(cp, ChatEvent{Type: EventTextChunk, SessionID: sessionID, Text: text})
			}
		}

	case "assistant":
		if msg, ok := event["message"].(map[string]any); ok {
			if content, ok := msg["content"].([]any); ok {
				for _, block := range content {
					blockMap, ok := block.(map[string]any)
					if !ok {
						continue
					}
					switch blockMap["type"] {
					case "tool_use":
						name, _ := blockMap["name"].(string)
						id, _ := blockMap["id"].(string)
						summary := summarizeToolInput(blockMap["input"])
						trySendEvent(cp, ChatEvent{
							Type: EventToolStart, SessionID: sessionID,
							ToolName: name, ToolID: id, ToolInput: summary,
						})
						trySendEvent(cp, ChatEvent{
							Type: EventStatus, SessionID: sessionID,
							Text: toolToStatusMessage(name, summary),
						})
					case "text":
						if text, ok := blockMap["text"].(string); ok && text != "" {
							trySendEvent(cp, ChatEvent{Type: EventTextChunk, SessionID: sessionID, Text: text})
						}
					}
				}
			}
		}

	case "user":
		if msg, ok := event["message"].(map[string]any); ok {
			if content, ok := msg["content"].([]any); ok {
				for _, block := range content {
					blockMap, ok := block.(map[string]any)
					if !ok || blockMap["type"] != "tool_result" {
						continue
					}
					toolUseID, _ := blockMap["tool_use_id"].(string)
					isError, _ := blockMap["is_error"].(bool)
					trySendEvent(cp, ChatEvent{
						Type: EventToolEnd, SessionID: sessionID,
						ToolID: toolUseID, ToolName: toolIDMap[toolUseID],
						ToolError: isError,
					})
				}
			}
		}

	case "result":
		costUSD, _ := event["total_cost_usd"].(float64)
		trySendEvent(cp, ChatEvent{
			Type: EventResult, SessionID: sessionID, CostUSD: costUSD,
		})

	case "error":
		errMsg := ""
		if errObj, ok := event["error"].(map[string]any); ok {
			errMsg, _ = errObj["message"].(string)
		} else if errStr, ok := event["error"].(string); ok {
			errMsg = errStr
		}
		if errMsg != "" {
			trySendEvent(cp, ChatEvent{Type: EventError, SessionID: sessionID, Text: errMsg})
		}
	}
}

// trySendEvent sends a ChatEvent to EventCh non-blocking.
func trySendEvent(cp *ClaudeProcess, ev ChatEvent) {
	if cp.EventCh == nil {
		return
	}
	select {
	case cp.EventCh <- ev:
	default:
	}
}

// toolToStatusMessage maps tool names to human-readable loading strings.
func toolToStatusMessage(toolName, args string) string {
	switch toolName {
	case "Read":
		return "Reading " + truncateStr(args, 60)
	case "Edit":
		return "Editing " + truncateStr(args, 60)
	case "Write":
		return "Writing " + truncateStr(args, 60)
	case "Bash":
		return "Running command..."
	case "Grep":
		return "Searching for " + truncateStr(args, 60)
	case "Glob":
		return "Finding files..."
	case "WebFetch":
		return "Fetching " + truncateStr(args, 60)
	case "WebSearch":
		return "Searching the web..."
	case "TodoWrite":
		return "Updating task list..."
	case "Task":
		return "Delegating to sub-agent..."
	default:
		return "Using " + toolName + "..."
	}
}

// truncateStr truncates a string to max length with "..." suffix.
func truncateStr(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max-3] + "..."
}

// extractChunkText returns the displayable text from a single stream-json event.
func extractChunkText(event map[string]any) string {
	switch event["type"] {
	case "content_block_delta":
		if delta, ok := event["delta"].(map[string]any); ok {
			if text, ok := delta["text"].(string); ok {
				return text
			}
		}
	case "assistant":
		if msg, ok := event["message"].(map[string]any); ok {
			var buf strings.Builder
			extractAssistantText(msg, &buf, nil)
			return buf.String()
		}
	}
	return ""
}

// buildArgs constructs the command-line arguments for interactive stream-json mode.
// Uses --permission-prompt-tool stdio so permission requests come as control_request
// events on stdout, and we respond via stdin control_response messages.
func buildArgs(opts SpawnOptions) []string {
	args := []string{
		"--output-format", "stream-json",
		"--input-format", "stream-json",
		"--verbose",
		"--permission-prompt-tool", "stdio",
	}

	if opts.SessionID != "" {
		if opts.Resume {
			args = append(args, "--resume", opts.SessionID)
		} else {
			args = append(args, "--session-id", opts.SessionID)
		}
	}

	if opts.Model != "" {
		args = append(args, "--model", opts.Model)
	}

	if len(opts.AllowedTools) > 0 {
		for _, tool := range opts.AllowedTools {
			args = append(args, "--allowedTools", tool)
		}
	}

	if opts.MaxBudget > 0 {
		args = append(args, "--max-budget-usd", fmt.Sprintf("%.2f", opts.MaxBudget))
	}

	if opts.SystemPrompt != "" {
		args = append(args, "--system-prompt", opts.SystemPrompt)
	}

	return args
}

// buildEnv merges the current process environment with any extra variables
// specified in the spawn options. CLAUDECODE and related vars are stripped so
// that claude does not refuse to start inside an existing Claude Code session.
func buildEnv(extra map[string]string) []string {
	skip := map[string]bool{
		"CLAUDECODE":                                true,
		"CLAUDE_CODE":                               true,
		"CLAUDE_CODE_SESSION_ID":                    true,
		"CLAUDE_CODE_ENTRYPOINT":                    true,
		"CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING": true,
		"CLAUDE_AGENT_SDK_VERSION":                  true,
	}
	env := make([]string, 0, len(os.Environ()))
	for _, kv := range os.Environ() {
		key := kv
		for i := 0; i < len(kv); i++ {
			if kv[i] == '=' {
				key = kv[:i]
				break
			}
		}
		if skip[key] {
			continue
		}
		env = append(env, kv)
	}
	for k, v := range extra {
		env = append(env, fmt.Sprintf("%s=%s", k, v))
	}
	// NO_COLOR + TERM=dumb to avoid ANSI escape sequences in output.
	env = append(env, "NO_COLOR=1", "TERM=dumb")
	return env
}

// parseStreamEvent extracts useful data from a single stream-json event line.
func parseStreamEvent(
	event map[string]any,
	responseText *strings.Builder,
	tokensIn *int64,
	tokensOut *int64,
	costUSD *float64,
	modelUsed *string,
	sessionID *string,
	toolIDMap map[string]string,
) {
	eventType, _ := event["type"].(string)

	switch eventType {
	case "assistant":
		if msg, ok := event["message"].(map[string]any); ok {
			extractAssistantText(msg, responseText, toolIDMap)
			if m, ok := msg["model"].(string); ok && m != "" {
				*modelUsed = m
			}
			extractUsage(msg, tokensIn, tokensOut)
		}

	case "user":
		if msg, ok := event["message"].(map[string]any); ok {
			if contentArr, ok := msg["content"].([]any); ok {
				for _, block := range contentArr {
					blockMap, ok := block.(map[string]any)
					if !ok || blockMap["type"] != "tool_result" {
						continue
					}
					toolUseID, _ := blockMap["tool_use_id"].(string)
					toolName := toolIDMap[toolUseID]
					if toolName == "" {
						continue
					}
					isError, _ := blockMap["is_error"].(bool)
					marker := "\u2713" // ✓
					if isError {
						marker = "\u2717" // ✗
					}
					fmt.Fprintf(responseText, "%s %s\n", marker, toolName)
				}
			}
		}

	case "content_block_delta":
		if delta, ok := event["delta"].(map[string]any); ok {
			if text, ok := delta["text"].(string); ok {
				responseText.WriteString(text)
			}
		}

	case "message_start":
		if msg, ok := event["message"].(map[string]any); ok {
			if m, ok := msg["model"].(string); ok && m != "" {
				*modelUsed = m
			}
			extractUsage(msg, tokensIn, tokensOut)
		}

	case "message_delta":
		if usage, ok := event["usage"].(map[string]any); ok {
			if v, ok := usage["input_tokens"].(float64); ok {
				*tokensIn = int64(v)
			}
			if v, ok := usage["output_tokens"].(float64); ok {
				*tokensOut = int64(v)
			}
		}

	case "result":
		extractResultEvent(event, responseText, tokensIn, tokensOut, costUSD, modelUsed, sessionID)

	case "error":
		// Claude emits error events (e.g., invalid session ID).
		if errMsg, ok := event["error"].(map[string]any); ok {
			if msg, ok := errMsg["message"].(string); ok && msg != "" {
				fmt.Fprintf(responseText, "[error]: %s\n", msg)
			}
		} else if errStr, ok := event["error"].(string); ok && errStr != "" {
			fmt.Fprintf(responseText, "[error]: %s\n", errStr)
		}

	case "system":
		// System messages (e.g. session init) — ignore.

	default:
		if text, ok := event["text"].(string); ok && text != "" {
			responseText.WriteString(text)
		}
		if content, ok := event["content"].(string); ok && content != "" {
			responseText.WriteString(content)
		}
	}
}

// extractAssistantText pulls text from the content blocks of an assistant message.
func extractAssistantText(msg map[string]any, buf *strings.Builder, toolIDMap map[string]string) {
	content, ok := msg["content"]
	if !ok {
		return
	}

	switch c := content.(type) {
	case string:
		buf.WriteString(c)
		buf.WriteByte('\n')
	case []any:
		for _, block := range c {
			blockMap, ok := block.(map[string]any)
			if !ok {
				continue
			}
			blockType, _ := blockMap["type"].(string)
			switch blockType {
			case "text":
				if text, ok := blockMap["text"].(string); ok && text != "" {
					buf.WriteString(text)
					buf.WriteByte('\n')
				}
			case "tool_use":
				toolName, _ := blockMap["name"].(string)
				if toolName == "" {
					continue
				}
				if id, ok := blockMap["id"].(string); ok && id != "" && toolIDMap != nil {
					toolIDMap[id] = toolName
				}
				argSummary := summarizeToolInput(blockMap["input"])
				if argSummary != "" {
					fmt.Fprintf(buf, "\u2699 %s: %s\n", toolName, argSummary)
				} else {
					fmt.Fprintf(buf, "\u2699 %s\n", toolName)
				}
			}
		}
	}
}

// summarizeToolInput returns a short human-readable summary of a tool_use input.
func summarizeToolInput(input any) string {
	m, ok := input.(map[string]any)
	if !ok {
		return ""
	}
	for _, key := range []string{"command", "file_path", "path", "query", "url", "pattern", "description", "content"} {
		if v, ok := m[key].(string); ok && v != "" {
			if len(v) > 80 {
				v = v[:77] + "..."
			}
			return v
		}
	}
	for _, v := range m {
		if s, ok := v.(string); ok && s != "" {
			if len(s) > 80 {
				s = s[:77] + "..."
			}
			return s
		}
	}
	return ""
}

// extractUsage pulls token counts from a usage map within a message.
func extractUsage(msg map[string]any, tokensIn, tokensOut *int64) {
	usage, ok := msg["usage"].(map[string]any)
	if !ok {
		return
	}
	if v, ok := usage["input_tokens"].(float64); ok {
		*tokensIn = int64(v)
	}
	if v, ok := usage["output_tokens"].(float64); ok {
		*tokensOut = int64(v)
	}
}

// extractResultEvent handles the "result" type event.
func extractResultEvent(
	event map[string]any,
	responseText *strings.Builder,
	tokensIn *int64,
	tokensOut *int64,
	costUSD *float64,
	modelUsed *string,
	sessionID *string,
) {
	if result, ok := event["result"].(string); ok && result != "" {
		if responseText.Len() == 0 {
			responseText.WriteString(result)
		}
	}

	if sid, ok := event["session_id"].(string); ok && sid != "" {
		*sessionID = sid
	}

	if c, ok := event["cost_usd"].(float64); ok {
		*costUSD = c
	}
	// Also check total_cost_usd (used in stream-json result events).
	if c, ok := event["total_cost_usd"].(float64); ok && c > 0 {
		*costUSD = c
	}

	if m, ok := event["model"].(string); ok && m != "" {
		*modelUsed = m
	}

	if usage, ok := event["usage"].(map[string]any); ok {
		if v, ok := usage["input_tokens"].(float64); ok {
			*tokensIn = int64(v)
		}
		if v, ok := usage["output_tokens"].(float64); ok {
			*tokensOut = int64(v)
		}
	}

	if resultObj, ok := event["result"].(map[string]any); ok {
		if text, ok := resultObj["text"].(string); ok && text != "" && responseText.Len() == 0 {
			responseText.WriteString(text)
		}
	}
}
