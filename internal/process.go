// Package internal contains the core logic for the bridge.claude plugin.
// It spawns Claude Code CLI processes, parses their stream-json output, and
// manages active session lifecycles.
package internal

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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
	PermissionMode string            // --permission-mode (optional)
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

// ClaudeProcess represents a running Claude Code CLI process.
type ClaudeProcess struct {
	SessionID string
	Cmd       *exec.Cmd
	StartedAt time.Time
	Done      chan struct{}

	mu       sync.Mutex
	exitErr  error
	finished bool
	response *ChatResponse // stored when process completes
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

// Spawn launches a Claude Code CLI process with the given options, reads its
// stream-json output, and returns a ChatResponse when the process exits. The
// caller's context controls cancellation -- if the context is cancelled, the
// spawned process is killed.
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

	// Capture stderr so we can include it in error messages.
	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	start := time.Now()

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("bridge.claude: start claude: %w", err)
	}

	// Parse stream-json output line by line.
	scanner := bufio.NewScanner(stdout)
	// Allow large lines (up to 1 MB) in case of big tool results.
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var (
		responseText strings.Builder
		tokensIn     int64
		tokensOut    int64
		costUSD      float64
		modelUsed    string
		sessionID    = opts.SessionID
	)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			// Not valid JSON; append as raw text.
			responseText.WriteString(line)
			responseText.WriteByte('\n')
			continue
		}

		parseStreamEvent(event, &responseText, &tokensIn, &tokensOut, &costUSD, &modelUsed, &sessionID)
	}

	waitErr := cmd.Wait()
	durationMs := time.Since(start).Milliseconds()

	// If the context was cancelled, report that rather than the generic exit error.
	if ctx.Err() != nil {
		return nil, fmt.Errorf("bridge.claude: process cancelled: %w", ctx.Err())
	}

	// A non-zero exit code is noteworthy but we still return whatever output
	// we captured. The caller can inspect ResponseText.
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

// SpawnAsync launches a Claude Code CLI process in the background and returns
// the ClaudeProcess handle immediately. The process result is NOT collected --
// this is used for persistent sessions where the caller may kill the process
// later.
func SpawnAsync(ctx context.Context, opts SpawnOptions) (*ClaudeProcess, *ChatResponse, error) {
	// For async spawn, we still run synchronously inside a wrapper so the
	// caller can get the response when the process completes. However, we
	// expose the ClaudeProcess for tracking and killing.
	args := buildArgs(opts)

	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Env = buildEnv(opts.Env)

	if opts.Workspace != "" {
		cmd.Dir = opts.Workspace
	}

	cp := &ClaudeProcess{
		SessionID: opts.SessionID,
		Cmd:       cmd,
		StartedAt: time.Now(),
		Done:      make(chan struct{}),
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("bridge.claude: stdout pipe: %w", err)
	}
	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	if err := cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("bridge.claude: start claude: %w", err)
	}

	// Read output synchronously, then mark done.
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var (
		responseText strings.Builder
		tokensIn     int64
		tokensOut    int64
		costUSD      float64
		modelUsed    string
		sessionID    = opts.SessionID
	)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			responseText.WriteString(line)
			responseText.WriteByte('\n')
			continue
		}

		parseStreamEvent(event, &responseText, &tokensIn, &tokensOut, &costUSD, &modelUsed, &sessionID)
	}

	waitErr := cmd.Wait()
	durationMs := time.Since(cp.StartedAt).Milliseconds()

	resp := &ChatResponse{
		ResponseText: strings.TrimSpace(responseText.String()),
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

// SpawnBackground launches a Claude Code CLI process and returns immediately.
// The process reads stdout and waits in a background goroutine. The caller can
// poll ClaudeProcess.GetResponse() or block with WaitResponse() to obtain the
// result once the process finishes.
func SpawnBackground(ctx context.Context, opts SpawnOptions) (*ClaudeProcess, error) {
	args := buildArgs(opts)
	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Env = buildEnv(opts.Env)
	if opts.Workspace != "" {
		cmd.Dir = opts.Workspace
	}

	cp := &ClaudeProcess{
		SessionID: opts.SessionID,
		Cmd:       cmd,
		StartedAt: time.Now(),
		Done:      make(chan struct{}),
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("bridge.claude: stdout pipe: %w", err)
	}
	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("bridge.claude: start claude: %w", err)
	}

	// Read output in background goroutine.
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
		)

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var event map[string]any
			if err := json.Unmarshal([]byte(line), &event); err != nil {
				responseText.WriteString(line)
				responseText.WriteByte('\n')
				continue
			}
			parseStreamEvent(event, &responseText, &tokensIn, &tokensOut, &costUSD, &modelUsed, &sessionID)
		}

		waitErr := cmd.Wait()
		durationMs := time.Since(cp.StartedAt).Milliseconds()

		resp := &ChatResponse{
			ResponseText: strings.TrimSpace(responseText.String()),
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
// text chunk as the response streams line by line. After the process exits,
// the full ChatResponse is returned. Used by the streaming tool handler.
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
	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	start := time.Now()

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("bridge.claude: start claude: %w", err)
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
	)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			responseText.WriteString(line)
			responseText.WriteByte('\n')
			continue
		}

		// Yield text chunk before accumulating into responseText.
		if chunk := extractChunkText(event); chunk != "" && chunkFn != nil {
			chunkFn([]byte(chunk))
		}

		parseStreamEvent(event, &responseText, &tokensIn, &tokensOut, &costUSD, &modelUsed, &sessionID)
	}

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

// extractChunkText returns the displayable text from a single stream-json event
// without modifying any accumulator. Used by SpawnStream to yield live chunks.
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
			extractAssistantText(msg, &buf)
			return buf.String()
		}
	}
	return ""
}

// buildArgs constructs the command-line arguments for claude -p.
func buildArgs(opts SpawnOptions) []string {
	args := []string{
		"-p", opts.Prompt,
		"--output-format", "stream-json",
		"--verbose",
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

	if opts.PermissionMode != "" {
		args = append(args, "--permission-mode", opts.PermissionMode)
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
	// Variables that make claude think it's running inside another Claude Code
	// session — claude refuses to start nested sessions unless these are unset.
	skip := map[string]bool{
		"CLAUDECODE":             true,
		"CLAUDE_CODE":            true,
		"CLAUDE_CODE_SESSION_ID": true,
		"CLAUDE_CODE_ENTRYPOINT": true,
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
	return env
}

// parseStreamEvent extracts useful data from a single stream-json event line.
// The Claude Code CLI stream-json format emits JSONL where each line has a
// "type" field. We look for assistant text content, usage statistics, and
// session metadata. If the format is unrecognized, we silently skip.
func parseStreamEvent(
	event map[string]any,
	responseText *strings.Builder,
	tokensIn *int64,
	tokensOut *int64,
	costUSD *float64,
	modelUsed *string,
	sessionID *string,
) {
	eventType, _ := event["type"].(string)

	switch eventType {
	case "assistant":
		// The "assistant" event contains a "message" field with the content.
		if msg, ok := event["message"].(map[string]any); ok {
			extractAssistantText(msg, responseText)
			// Model may appear here.
			if m, ok := msg["model"].(string); ok && m != "" {
				*modelUsed = m
			}
			// Usage may appear in the message.
			extractUsage(msg, tokensIn, tokensOut)
		}

	case "content_block_delta":
		// Partial text deltas during streaming.
		if delta, ok := event["delta"].(map[string]any); ok {
			if text, ok := delta["text"].(string); ok {
				responseText.WriteString(text)
			}
		}

	case "message_start":
		// May contain the model and session info.
		if msg, ok := event["message"].(map[string]any); ok {
			if m, ok := msg["model"].(string); ok && m != "" {
				*modelUsed = m
			}
			extractUsage(msg, tokensIn, tokensOut)
		}

	case "message_delta":
		// Final usage stats often appear here.
		if usage, ok := event["usage"].(map[string]any); ok {
			if v, ok := usage["input_tokens"].(float64); ok {
				*tokensIn = int64(v)
			}
			if v, ok := usage["output_tokens"].(float64); ok {
				*tokensOut = int64(v)
			}
		}

	case "result":
		// The final result event from Claude Code CLI.
		extractResultEvent(event, responseText, tokensIn, tokensOut, costUSD, modelUsed, sessionID)

	default:
		// For unknown event types, check for common fields.
		// Some events carry text directly.
		if text, ok := event["text"].(string); ok && text != "" {
			responseText.WriteString(text)
		}
		// Check for a "content" field.
		if content, ok := event["content"].(string); ok && content != "" {
			responseText.WriteString(content)
		}
	}
}

// extractAssistantText pulls text from the content blocks of an assistant message.
func extractAssistantText(msg map[string]any, buf *strings.Builder) {
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
			if blockType == "text" {
				if text, ok := blockMap["text"].(string); ok {
					buf.WriteString(text)
					buf.WriteByte('\n')
				}
			}
		}
	}
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

// extractResultEvent handles the "result" type event which Claude Code CLI
// emits as the final summary.
func extractResultEvent(
	event map[string]any,
	responseText *strings.Builder,
	tokensIn *int64,
	tokensOut *int64,
	costUSD *float64,
	modelUsed *string,
	sessionID *string,
) {
	// Result text.
	if result, ok := event["result"].(string); ok && result != "" {
		// Only use result text if we haven't collected anything yet.
		if responseText.Len() == 0 {
			responseText.WriteString(result)
		}
	}

	// Session ID from result.
	if sid, ok := event["session_id"].(string); ok && sid != "" {
		*sessionID = sid
	}

	// Cost.
	if c, ok := event["cost_usd"].(float64); ok {
		*costUSD = c
	}

	// Duration is handled by the caller via wall-clock timing.

	// Model.
	if m, ok := event["model"].(string); ok && m != "" {
		*modelUsed = m
	}

	// Usage stats in result event.
	if usage, ok := event["usage"].(map[string]any); ok {
		if v, ok := usage["input_tokens"].(float64); ok {
			*tokensIn = int64(v)
		}
		if v, ok := usage["output_tokens"].(float64); ok {
			*tokensOut = int64(v)
		}
	}

	// Sometimes the result has a nested "result" object with text.
	if resultObj, ok := event["result"].(map[string]any); ok {
		if text, ok := resultObj["text"].(string); ok && text != "" && responseText.Len() == 0 {
			responseText.WriteString(text)
		}
	}
}
