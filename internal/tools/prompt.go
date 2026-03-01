// Package tools contains the tool schemas and handler functions for the
// bridge.claude plugin. Each exported function pair (Schema + Handler) follows
// the same pattern used across all Orchestra plugins.
package tools

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	pluginv1 "github.com/orchestra-mcp/gen-go/orchestra/plugin/v1"
	"github.com/orchestra-mcp/sdk-go/helpers"
	"github.com/orchestra-mcp/sdk-go/plugin"
	"google.golang.org/protobuf/types/known/structpb"
)

// ToolHandler is an alias for readability.
type ToolHandler = func(ctx context.Context, req *pluginv1.ToolRequest) (*pluginv1.ToolResponse, error)

// BridgePluginInterface defines the methods the tools package needs from the
// BridgePlugin. This avoids a circular import between internal and tools.
type BridgePluginInterface interface {
	TrackProcess(proc ProcessHandle)
	GetProcess(sessionID string) ProcessHandle
	RemoveProcess(sessionID string) ProcessHandle
	ListProcesses() []ProcessHandle
}

// ProcessHandle is the interface that tools use to interact with a running
// ClaudeProcess, avoiding direct type dependency on internal.ClaudeProcess.
type ProcessHandle interface {
	IsRunning() bool
	GetSessionID() string
	GetPID() int
	GetStartedAt() string
	GetUptimeSeconds() float64
	Kill() error
	GetResponse() *ChatResponse
	WaitResponse(ctx context.Context) (*ChatResponse, error)
}

// SpawnFunc is the function signature for spawning a Claude Code CLI process.
// This is injected to avoid circular imports.
type SpawnFunc func(ctx context.Context, opts SpawnOptions) (*ChatResponse, error)

// SpawnAsyncFunc is the function signature for spawning an async Claude process.
type SpawnAsyncFunc func(ctx context.Context, opts SpawnOptions) (ProcessHandle, *ChatResponse, error)

// SpawnBackgroundFunc is the function signature for spawning a background
// Claude process that returns immediately without waiting for completion.
type SpawnBackgroundFunc func(ctx context.Context, opts SpawnOptions) (ProcessHandle, error)

// SpawnStreamFunc is the function signature for streaming spawn. chunkFn is
// called for each text chunk as it is emitted by the Claude process.
type SpawnStreamFunc func(ctx context.Context, opts SpawnOptions, chunkFn func([]byte)) (*ChatResponse, error)

// SpawnOptions mirrors the internal SpawnOptions for use by tool handlers.
type SpawnOptions struct {
	SessionID      string
	Resume         bool
	Prompt         string
	Model          string
	Workspace      string
	AllowedTools   []string
	PermissionMode string
	MaxBudget      float64
	SystemPrompt   string
	Env            map[string]string
}

// ChatResponse mirrors the internal ChatResponse for use by tool handlers.
type ChatResponse struct {
	ResponseText string  `json:"response_text"`
	TokensIn     int64   `json:"tokens_in"`
	TokensOut    int64   `json:"tokens_out"`
	CostUSD      float64 `json:"cost_usd"`
	ModelUsed    string  `json:"model_used"`
	DurationMs   int64   `json:"duration_ms"`
	SessionID    string  `json:"session_id"`
}

// Bridge holds the injected dependencies that tool handlers need.
type Bridge struct {
	Spawn           SpawnFunc
	SpawnAsync      SpawnAsyncFunc
	SpawnBackground SpawnBackgroundFunc
	SpawnStream     SpawnStreamFunc
	Plugin          BridgePluginInterface
}

// --- ai_prompt ---

// AIPromptSchema returns the JSON Schema for the ai_prompt tool.
func AIPromptSchema() *structpb.Struct {
	s, _ := structpb.NewStruct(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"prompt": map[string]any{
				"type":        "string",
				"description": "The prompt to send to Claude Code",
			},
			"model": map[string]any{
				"type":        "string",
				"description": "Model to use (e.g., sonnet, opus, haiku)",
			},
			"workspace": map[string]any{
				"type":        "string",
				"description": "Working directory for Claude Code",
			},
			"allowed_tools": map[string]any{
				"type":        "string",
				"description": "Comma-separated list of allowed tools (e.g., Bash,Read,Edit)",
			},
			"permission_mode": map[string]any{
				"type":        "string",
				"description": "Permission mode (e.g., default, plan, bypassPermissions)",
			},
			"max_budget": map[string]any{
				"type":        "number",
				"description": "Maximum budget in USD",
			},
			"system_prompt": map[string]any{
				"type":        "string",
				"description": "Custom system prompt",
			},
			"env": map[string]any{
				"type":        "string",
				"description": "JSON object of environment variables (e.g., {\"ANTHROPIC_API_KEY\": \"sk-...\"})",
			},
			"wait": map[string]any{
				"type":        "boolean",
				"description": "Wait for completion and return full response (default: false). Set to true for synchronous behavior.",
			},
		},
		"required": []any{"prompt"},
	})
	return s
}

// AIPrompt returns a tool handler that sends a one-shot prompt to Claude Code
// CLI. By default the process runs in the background and the caller polls via
// session_status. Set wait=true for synchronous (blocking) behavior.
func AIPrompt(bridge *Bridge) plugin.ToolHandler {
	return func(ctx context.Context, req *pluginv1.ToolRequest) (*pluginv1.ToolResponse, error) {
		if err := helpers.ValidateRequired(req.Arguments, "prompt"); err != nil {
			return helpers.ErrorResult("validation_error", err.Error()), nil
		}

		opts, err := parseCommonOpts(req.Arguments)
		if err != nil {
			return helpers.ErrorResult("validation_error", err.Error()), nil
		}
		// One-shot: no session ID.
		opts.SessionID = ""
		opts.Resume = false

		wait := helpers.GetBool(req.Arguments, "wait")

		// Synchronous mode: block until completion.
		if wait || bridge.SpawnBackground == nil {
			resp, err := bridge.Spawn(ctx, opts)
			if err != nil {
				return helpers.ErrorResult("spawn_error", err.Error()), nil
			}
			return helpers.TextResult(formatChatResponse(resp)), nil
		}

		// Async mode (default): start in background, return immediately.
		sessionID := generateSessionID()
		opts.SessionID = sessionID

		proc, err := bridge.SpawnBackground(ctx, opts)
		if err != nil {
			return helpers.ErrorResult("spawn_error", err.Error()), nil
		}

		bridge.Plugin.TrackProcess(proc)

		return helpers.TextResult(fmt.Sprintf(
			"## Prompt Started\n\n"+
				"- **Session:** %s\n"+
				"- **PID:** %d\n"+
				"- **Status:** running\n\n"+
				"Use `session_status` with session_id `%s` to check progress.\n"+
				"The response will be available when the process completes.\n",
			sessionID, proc.GetPID(), sessionID,
		)), nil
	}
}

// generateSessionID creates a random hex session ID for background prompts.
func generateSessionID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return fmt.Sprintf("prompt-%x", b)
}

// --- Common helpers ---

// parseCommonOpts extracts the shared spawn options from tool arguments.
func parseCommonOpts(args *structpb.Struct) (SpawnOptions, error) {
	prompt := helpers.GetString(args, "prompt")
	model := helpers.GetString(args, "model")
	workspace := helpers.GetString(args, "workspace")
	allowedToolsRaw := helpers.GetString(args, "allowed_tools")
	permissionMode := helpers.GetString(args, "permission_mode")
	maxBudget := helpers.GetFloat64(args, "max_budget")
	systemPrompt := helpers.GetString(args, "system_prompt")
	envRaw := helpers.GetString(args, "env")

	// Default workspace to current working directory.
	if workspace == "" {
		cwd, err := os.Getwd()
		if err == nil {
			workspace = cwd
		}
	}

	// Parse allowed tools from comma-separated string.
	var allowedTools []string
	if allowedToolsRaw != "" {
		for _, t := range strings.Split(allowedToolsRaw, ",") {
			t = strings.TrimSpace(t)
			if t != "" {
				allowedTools = append(allowedTools, t)
			}
		}
	}

	// Parse env from JSON string.
	var envMap map[string]string
	if envRaw != "" {
		if err := json.Unmarshal([]byte(envRaw), &envMap); err != nil {
			return SpawnOptions{}, fmt.Errorf("invalid env JSON: %w", err)
		}
	}

	return SpawnOptions{
		Prompt:         prompt,
		Model:          model,
		Workspace:      workspace,
		AllowedTools:   allowedTools,
		PermissionMode: permissionMode,
		MaxBudget:      maxBudget,
		SystemPrompt:   systemPrompt,
		Env:            envMap,
	}, nil
}

// AIPromptStreamSchema returns the JSON Schema for the ai_prompt_stream tool.
// It mirrors ai_prompt but is registered as a streaming tool.
func AIPromptStreamSchema() *structpb.Struct {
	return AIPromptSchema()
}

// AIPromptStream returns a StreamingToolHandler that yields text chunks as they
// are emitted by the Claude Code CLI. Each chunk is sent as a StreamChunk to
// the caller; a final StreamEnd is sent when the process exits.
func AIPromptStream(bridge *Bridge) plugin.StreamingToolHandler {
	return func(ctx context.Context, req *pluginv1.StreamStart, chunks chan<- []byte) error {
		if req.Arguments == nil {
			return fmt.Errorf("missing required parameter: prompt")
		}
		promptVal := req.Arguments.GetFields()["prompt"]
		if promptVal == nil || promptVal.GetStringValue() == "" {
			return fmt.Errorf("missing required parameter: prompt")
		}

		opts, err := parseCommonOpts(req.Arguments)
		if err != nil {
			return err
		}
		opts.SessionID = ""
		opts.Resume = false

		if bridge.SpawnStream == nil {
			return fmt.Errorf("streaming not supported: SpawnStream not configured")
		}

		_, err = bridge.SpawnStream(ctx, opts, func(chunk []byte) {
			select {
			case chunks <- chunk:
			case <-ctx.Done():
			}
		})
		return err
	}
}

// formatChatResponse formats a ChatResponse as a Markdown string for display.
func formatChatResponse(resp *ChatResponse) string {
	var b strings.Builder
	b.WriteString(resp.ResponseText)
	b.WriteString("\n\n---\n")

	if resp.SessionID != "" {
		fmt.Fprintf(&b, "- **Session:** %s\n", resp.SessionID)
	}
	if resp.ModelUsed != "" {
		fmt.Fprintf(&b, "- **Model:** %s\n", resp.ModelUsed)
	}
	if resp.TokensIn > 0 || resp.TokensOut > 0 {
		fmt.Fprintf(&b, "- **Tokens:** %d in / %d out\n", resp.TokensIn, resp.TokensOut)
	}
	if resp.CostUSD > 0 {
		fmt.Fprintf(&b, "- **Cost:** $%.4f\n", resp.CostUSD)
	}
	if resp.DurationMs > 0 {
		fmt.Fprintf(&b, "- **Duration:** %dms\n", resp.DurationMs)
	}

	return b.String()
}
