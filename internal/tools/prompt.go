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
	SetSessionID(string)
	GetPID() int
	GetStartedAt() string
	GetUptimeSeconds() float64
	Kill() error
	GetResponse() *ChatResponse
	WaitResponse(ctx context.Context) (*ChatResponse, error)
	SetAutoApprove(bool)
	GetEventCh() <-chan ChatEvent
}

// ChatEventType classifies streaming events for the UI.
type ChatEventType string

const (
	EventTextChunk  ChatEventType = "text_chunk"
	EventToolStart  ChatEventType = "tool_start"
	EventToolEnd    ChatEventType = "tool_end"
	EventThinking   ChatEventType = "thinking"
	EventStatus     ChatEventType = "status"
	EventResult     ChatEventType = "result"
	EventError      ChatEventType = "error"
	EventPermission ChatEventType = "permission" // permission request from Claude
	EventQuestion   ChatEventType = "question"   // AskUserQuestion from Claude
)

// ChatEvent is a single granular event emitted during a Claude session.
type ChatEvent struct {
	Type       ChatEventType `json:"type"`
	SessionID  string        `json:"session_id"`
	Text       string        `json:"text,omitempty"`
	ToolName   string        `json:"tool_name,omitempty"`
	ToolID     string        `json:"tool_id,omitempty"`
	ToolInput  string        `json:"tool_input,omitempty"`
	ToolData   string        `json:"tool_data,omitempty"`   // Full JSON input for rich rendering
	ToolResult string        `json:"tool_result,omitempty"` // Tool output/result content
	ToolError  bool          `json:"tool_error,omitempty"`
	TokensIn   int64         `json:"tokens_in,omitempty"`
	TokensOut  int64         `json:"tokens_out,omitempty"`
	CostUSD    float64       `json:"cost_usd,omitempty"`
	ModelUsed  string        `json:"model_used,omitempty"`
	DurationMs int64         `json:"duration_ms,omitempty"`
	// Permission/Question fields
	RequestID string `json:"request_id,omitempty"` // control_request ID for responding
	Reason    string `json:"reason,omitempty"`     // why the tool needs permission
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
	ResponseText string      `json:"response_text"`
	TokensIn     int64       `json:"tokens_in"`
	TokensOut    int64       `json:"tokens_out"`
	CostUSD      float64     `json:"cost_usd"`
	ModelUsed    string      `json:"model_used"`
	DurationMs   int64       `json:"duration_ms"`
	SessionID    string      `json:"session_id"`
	ToolEvents   []ChatEvent `json:"tool_events,omitempty"`
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
		// One-shot: don't pass --session-id to Claude CLI (it requires UUIDs).
		// We generate an internal tracking ID but keep opts.SessionID empty
		// so buildArgs won't add --session-id to the CLI invocation.
		opts.SessionID = ""
		opts.Resume = false

		wait := helpers.GetBool(req.Arguments, "wait")

		// Generate an internal tracking ID (not passed to Claude CLI).
		trackingID := generateSessionID()

		// Spawn without --session-id (one-shot mode).
		proc, err := bridge.SpawnBackground(ctx, opts)
		if err != nil {
			return helpers.ErrorResult("spawn_error", err.Error()), nil
		}

		// When permission_mode is bypassPermissions, auto-approve all
		// permission requests on the Go side immediately.
		if opts.PermissionMode == "bypassPermissions" || opts.PermissionMode == "dontAsk" {
			proc.SetAutoApprove(true)
		}

		// Set tracking ID on the process for internal lookups.
		proc.SetSessionID(trackingID)
		bridge.Plugin.TrackProcess(proc)

		// Synchronous mode: wait for completion before returning.
		// Do NOT auto-approve — permission requests go to PermissionCh
		// and the Swift UI drains them via get_pending_permission polling.
		if wait {
			resp, waitErr := proc.WaitResponse(ctx)
			if waitErr != nil && resp == nil {
				return helpers.ErrorResult("spawn_error", waitErr.Error()), nil
			}
			if resp != nil {
				return helpers.TextResult(formatChatResponse(resp)), nil
			}
			return helpers.TextResult("[no response]"), nil
		}

		return helpers.TextResult(fmt.Sprintf(
			"## Prompt Started\n\n"+
				"- **Session:** %s\n"+
				"- **PID:** %d\n"+
				"- **Status:** running\n\n"+
				"Use `session_status` with session_id `%s` to check progress.\n"+
				"The response will be available when the process completes.\n",
			proc.GetSessionID(), proc.GetPID(), proc.GetSessionID(),
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

	// Default workspace to the orchestra project root (set by the serve
	// command), falling back to the current working directory.
	if workspace == "" {
		workspace = os.Getenv("ORCHESTRA_WORKSPACE")
		if workspace == "" {
			cwd, err := os.Getwd()
			if err == nil {
				workspace = cwd
			}
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

// --- chat_stream ---

// ChatStreamSchema returns the JSON Schema for the chat_stream streaming tool.
// It accepts session_id + prompt for session-aware streaming.
func ChatStreamSchema() *structpb.Struct {
	s, _ := structpb.NewStruct(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"session_id": map[string]any{
				"type":        "string",
				"description": "Session ID for the chat session",
			},
			"prompt": map[string]any{
				"type":        "string",
				"description": "The prompt to send to Claude Code",
			},
			"resume": map[string]any{
				"type":        "boolean",
				"description": "Whether to resume an existing Claude session",
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
				"description": "JSON object of environment variables",
			},
		},
		"required": []any{"session_id", "prompt"},
	})
	return s
}

// ChatStream returns a StreamingToolHandler that yields ChatEvent JSON chunks
// as they are emitted by the Claude Code CLI. The caller receives granular
// events (text_chunk, tool_start, tool_end, status, result, error) and can
// update the UI in real-time.
func ChatStream(bridge *Bridge) plugin.StreamingToolHandler {
	return func(ctx context.Context, req *pluginv1.StreamStart, chunks chan<- []byte) error {
		if req.Arguments == nil {
			return fmt.Errorf("missing required parameters: session_id, prompt")
		}

		sessionID := req.Arguments.GetFields()["session_id"]
		if sessionID == nil || sessionID.GetStringValue() == "" {
			return fmt.Errorf("missing required parameter: session_id")
		}
		promptVal := req.Arguments.GetFields()["prompt"]
		if promptVal == nil || promptVal.GetStringValue() == "" {
			return fmt.Errorf("missing required parameter: prompt")
		}

		opts, err := parseCommonOpts(req.Arguments)
		if err != nil {
			return err
		}
		opts.SessionID = sessionID.GetStringValue()

		// Check if we should resume an existing Claude session.
		if resumeVal := req.Arguments.GetFields()["resume"]; resumeVal != nil {
			opts.Resume = resumeVal.GetBoolValue()
		}

		if bridge.SpawnBackground == nil {
			return fmt.Errorf("streaming not supported: SpawnBackground not configured")
		}

		proc, err := bridge.SpawnBackground(ctx, opts)
		if err != nil {
			return err
		}

		// When permission_mode is bypassPermissions, auto-approve all
		// permission requests on the Go side immediately.
		if opts.PermissionMode == "bypassPermissions" || opts.PermissionMode == "dontAsk" {
			proc.SetAutoApprove(true)
		}

		bridge.Plugin.TrackProcess(proc)
		// For non-bypass modes, permission events flow through EventCh
		// and the Swift UI presents them. User responds via respond_to_permission.

		// Drain EventCh and send each event as a JSON chunk.
		eventCh := proc.GetEventCh()
		if eventCh == nil {
			// Fallback: wait for completion and send a single result event.
			resp, waitErr := proc.WaitResponse(ctx)
			if waitErr != nil && resp == nil {
				return waitErr
			}
			if resp != nil {
				ev := ChatEvent{
					Type:       EventResult,
					SessionID:  opts.SessionID,
					Text:       resp.ResponseText,
					TokensIn:   resp.TokensIn,
					TokensOut:  resp.TokensOut,
					CostUSD:    resp.CostUSD,
					ModelUsed:  resp.ModelUsed,
					DurationMs: resp.DurationMs,
				}
				data, _ := json.Marshal(ev)
				select {
				case chunks <- data:
				case <-ctx.Done():
				}
			}
			return nil
		}

		// Stream events until the channel closes or context is cancelled.
		for {
			select {
			case ev, ok := <-eventCh:
				if !ok {
					return nil // EventCh closed — process done
				}
				data, marshalErr := json.Marshal(ev)
				if marshalErr != nil {
					continue
				}
				select {
				case chunks <- data:
				case <-ctx.Done():
					return ctx.Err()
				}
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
}

// formatChatResponse returns the ChatResponse as a JSON string with clean
// structured fields. Callers (send_message, ai_prompt) parse the JSON to
// extract response text and metadata separately — no regex stripping needed.
func formatChatResponse(resp *ChatResponse) string {
	data := map[string]any{
		"response":    resp.ResponseText,
		"session_id":  resp.SessionID,
		"model":       resp.ModelUsed,
		"tokens_in":   resp.TokensIn,
		"tokens_out":  resp.TokensOut,
		"cost_usd":    resp.CostUSD,
		"duration_ms": resp.DurationMs,
	}
	if len(resp.ToolEvents) > 0 {
		data["tool_events"] = resp.ToolEvents
	}
	raw, err := json.Marshal(data)
	if err != nil {
		return resp.ResponseText // fallback to raw text
	}
	return string(raw)
}
