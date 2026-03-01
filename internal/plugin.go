package internal

import (
	"context"
	"os"
	"sync"

	"github.com/orchestra-mcp/sdk-go/plugin"
	"github.com/orchestra-mcp/plugin-bridge-claude/internal/tools"
)

// BridgePlugin manages Claude Code CLI processes and registers all bridge tools.
type BridgePlugin struct {
	active map[string]*ClaudeProcess
	mu     sync.RWMutex
}

// NewBridgePlugin creates a new BridgePlugin with an empty active process map.
func NewBridgePlugin() *BridgePlugin {
	return &BridgePlugin{
		active: make(map[string]*ClaudeProcess),
	}
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

	// --- Prompt tools (2) ---
	builder.RegisterTool("ai_prompt",
		"Send a one-shot prompt to Claude Code CLI and return the response",
		tools.AIPromptSchema(), tools.AIPrompt(bridge))

	builder.RegisterStreamingTool("ai_prompt_stream",
		"Stream a one-shot prompt to Claude Code CLI, yielding text chunks as they arrive",
		tools.AIPromptStreamSchema(), tools.AIPromptStream(bridge))

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
}

// spawnAdapter converts from tools.SpawnOptions to internal SpawnOptions and
// calls the internal Spawn function.
func (bp *BridgePlugin) spawnAdapter(ctx context.Context, opts tools.SpawnOptions) (*tools.ChatResponse, error) {
	internalOpts := convertOpts(opts)
	resp, err := Spawn(ctx, internalOpts)
	if err != nil {
		return nil, err
	}
	return convertResp(resp), nil
}

// spawnAsyncAdapter converts from tools.SpawnOptions and calls SpawnAsync,
// returning the process handle and response.
func (bp *BridgePlugin) spawnAsyncAdapter(ctx context.Context, opts tools.SpawnOptions) (tools.ProcessHandle, *tools.ChatResponse, error) {
	internalOpts := convertOpts(opts)
	proc, resp, err := SpawnAsync(ctx, internalOpts)
	if err != nil {
		return nil, nil, err
	}
	return wrapProcess(proc), convertResp(resp), nil
}

// spawnBackgroundAdapter converts from tools.SpawnOptions and calls
// SpawnBackground, returning the process handle immediately.
func (bp *BridgePlugin) spawnBackgroundAdapter(ctx context.Context, opts tools.SpawnOptions) (tools.ProcessHandle, error) {
	internalOpts := convertOpts(opts)
	proc, err := SpawnBackground(ctx, internalOpts)
	if err != nil {
		return nil, err
	}
	return wrapProcess(proc), nil
}

// spawnStreamAdapter converts from tools.SpawnOptions and calls SpawnStream,
// invoking chunkFn for each text chunk emitted by the Claude process.
func (bp *BridgePlugin) spawnStreamAdapter(ctx context.Context, opts tools.SpawnOptions, chunkFn func([]byte)) (*tools.ChatResponse, error) {
	internalOpts := convertOpts(opts)
	resp, err := SpawnStream(ctx, internalOpts, chunkFn)
	if err != nil {
		return nil, err
	}
	return convertResp(resp), nil
}

// --- BridgePluginInterface implementation ---

// TrackProcess adds a process to the active map.
func (bp *BridgePlugin) TrackProcess(proc tools.ProcessHandle) {
	cp := unwrapProcess(proc)
	if cp == nil {
		return
	}
	bp.mu.Lock()
	defer bp.mu.Unlock()
	bp.active[cp.SessionID] = cp
}

// GetProcess returns the process for the given session ID, or nil if not found.
func (bp *BridgePlugin) GetProcess(sessionID string) tools.ProcessHandle {
	bp.mu.RLock()
	defer bp.mu.RUnlock()
	proc, ok := bp.active[sessionID]
	if !ok {
		return nil
	}
	return wrapProcess(proc)
}

// RemoveProcess removes and returns the process for the given session ID.
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

// ListProcesses returns a snapshot of all active processes.
func (bp *BridgePlugin) ListProcesses() []tools.ProcessHandle {
	bp.mu.RLock()
	defer bp.mu.RUnlock()
	procs := make([]tools.ProcessHandle, 0, len(bp.active))
	for _, proc := range bp.active {
		procs = append(procs, wrapProcess(proc))
	}
	return procs
}

// KillAll terminates all active Claude Code CLI processes. Called during
// shutdown to ensure no orphaned processes remain.
func (bp *BridgePlugin) KillAll() {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	for id, proc := range bp.active {
		if proc.Cmd != nil && proc.Cmd.Process != nil {
			_ = proc.Cmd.Process.Signal(os.Kill)
		}
		delete(bp.active, id)
	}
}

// --- Process adapter ---

// processAdapter wraps a *ClaudeProcess to satisfy tools.ProcessHandle,
// converting internal types (e.g., *ChatResponse) to tools types.
type processAdapter struct {
	cp *ClaudeProcess
}

func (a *processAdapter) IsRunning() bool           { return a.cp.IsRunning() }
func (a *processAdapter) GetSessionID() string       { return a.cp.GetSessionID() }
func (a *processAdapter) GetPID() int                { return a.cp.GetPID() }
func (a *processAdapter) GetStartedAt() string       { return a.cp.GetStartedAt() }
func (a *processAdapter) GetUptimeSeconds() float64  { return a.cp.GetUptimeSeconds() }
func (a *processAdapter) Kill() error                { return a.cp.Kill() }

func (a *processAdapter) GetResponse() *tools.ChatResponse {
	resp := a.cp.GetResponse()
	return convertResp(resp)
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

// unwrapProcess extracts the underlying *ClaudeProcess from a ProcessHandle.
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
