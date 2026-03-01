package tools

import (
	"context"
	"fmt"
	"strings"

	pluginv1 "github.com/orchestra-mcp/gen-go/orchestra/plugin/v1"
	"github.com/orchestra-mcp/sdk-go/helpers"
	"github.com/orchestra-mcp/sdk-go/plugin"
	"google.golang.org/protobuf/types/known/structpb"
)

// --- spawn_session ---

// SpawnSessionSchema returns the JSON Schema for the spawn_session tool.
func SpawnSessionSchema() *structpb.Struct {
	s, _ := structpb.NewStruct(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"session_id": map[string]any{
				"type":        "string",
				"description": "Session UUID",
			},
			"prompt": map[string]any{
				"type":        "string",
				"description": "The prompt to send",
			},
			"resume": map[string]any{
				"type":        "boolean",
				"description": "Resume existing session (default: false)",
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
		"required": []any{"session_id", "prompt"},
	})
	return s
}

// SpawnSession returns a tool handler that spawns a persistent Claude Code CLI
// session. By default the process runs in the background and the caller polls
// via session_status. Set wait=true for synchronous (blocking) behavior, which
// is used by cross-plugin callers like send_message.
func SpawnSession(bridge *Bridge) plugin.ToolHandler {
	return func(ctx context.Context, req *pluginv1.ToolRequest) (*pluginv1.ToolResponse, error) {
		if err := helpers.ValidateRequired(req.Arguments, "session_id", "prompt"); err != nil {
			return helpers.ErrorResult("validation_error", err.Error()), nil
		}

		sessionID := helpers.GetString(req.Arguments, "session_id")
		resume := helpers.GetBool(req.Arguments, "resume")
		wait := helpers.GetBool(req.Arguments, "wait")

		opts, err := parseCommonOpts(req.Arguments)
		if err != nil {
			return helpers.ErrorResult("validation_error", err.Error()), nil
		}
		opts.SessionID = sessionID
		opts.Resume = resume

		// Synchronous mode: block until completion (used by cross-plugin
		// callers like send_message which have the full QUIC timeout).
		if wait || bridge.SpawnBackground == nil {
			proc, resp, err := bridge.SpawnAsync(ctx, opts)
			if err != nil {
				return helpers.ErrorResult("spawn_error", err.Error()), nil
			}
			bridge.Plugin.TrackProcess(proc)
			return helpers.TextResult(formatChatResponse(resp)), nil
		}

		// Async mode (default): start in background, return immediately.
		proc, err := bridge.SpawnBackground(ctx, opts)
		if err != nil {
			return helpers.ErrorResult("spawn_error", err.Error()), nil
		}

		bridge.Plugin.TrackProcess(proc)

		return helpers.TextResult(fmt.Sprintf(
			"## Session Started\n\n"+
				"- **Session:** %s\n"+
				"- **PID:** %d\n"+
				"- **Status:** running\n\n"+
				"Use `session_status` with session_id `%s` to check progress.\n",
			sessionID, proc.GetPID(), sessionID,
		)), nil
	}
}

// --- kill_session ---

// KillSessionSchema returns the JSON Schema for the kill_session tool.
func KillSessionSchema() *structpb.Struct {
	s, _ := structpb.NewStruct(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"session_id": map[string]any{
				"type":        "string",
				"description": "Session UUID to kill",
			},
		},
		"required": []any{"session_id"},
	})
	return s
}

// KillSession returns a tool handler that kills a running Claude Code CLI
// session and removes it from the active process map.
func KillSession(bridge *Bridge) plugin.ToolHandler {
	return func(ctx context.Context, req *pluginv1.ToolRequest) (*pluginv1.ToolResponse, error) {
		if err := helpers.ValidateRequired(req.Arguments, "session_id"); err != nil {
			return helpers.ErrorResult("validation_error", err.Error()), nil
		}

		sessionID := helpers.GetString(req.Arguments, "session_id")

		proc := bridge.Plugin.RemoveProcess(sessionID)
		if proc == nil {
			return helpers.ErrorResult("not_found",
				fmt.Sprintf("no active session found with ID %q", sessionID)), nil
		}

		if proc.IsRunning() {
			if err := proc.Kill(); err != nil {
				return helpers.ErrorResult("kill_error",
					fmt.Sprintf("failed to kill session %s: %v", sessionID, err)), nil
			}
			return helpers.TextResult(
				fmt.Sprintf("Killed session **%s** (PID %d)", sessionID, proc.GetPID())), nil
		}

		return helpers.TextResult(
			fmt.Sprintf("Session **%s** was already finished; removed from active list", sessionID)), nil
	}
}

// --- session_status ---

// SessionStatusSchema returns the JSON Schema for the session_status tool.
func SessionStatusSchema() *structpb.Struct {
	s, _ := structpb.NewStruct(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"session_id": map[string]any{
				"type":        "string",
				"description": "Session UUID to check",
			},
		},
		"required": []any{"session_id"},
	})
	return s
}

// SessionStatus returns a tool handler that reports the current status of a
// Claude Code CLI session. When the process has finished, the response is
// included in the output.
func SessionStatus(bridge *Bridge) plugin.ToolHandler {
	return func(ctx context.Context, req *pluginv1.ToolRequest) (*pluginv1.ToolResponse, error) {
		if err := helpers.ValidateRequired(req.Arguments, "session_id"); err != nil {
			return helpers.ErrorResult("validation_error", err.Error()), nil
		}

		sessionID := helpers.GetString(req.Arguments, "session_id")

		proc := bridge.Plugin.GetProcess(sessionID)
		if proc == nil {
			return helpers.ErrorResult("not_found",
				fmt.Sprintf("no session found with ID %q", sessionID)), nil
		}

		var b strings.Builder
		fmt.Fprintf(&b, "## Session: %s\n\n", sessionID)

		if proc.IsRunning() {
			fmt.Fprintf(&b, "- **Status:** running\n")
		} else {
			fmt.Fprintf(&b, "- **Status:** finished\n")
		}

		fmt.Fprintf(&b, "- **PID:** %d\n", proc.GetPID())
		fmt.Fprintf(&b, "- **Started:** %s\n", proc.GetStartedAt())
		fmt.Fprintf(&b, "- **Uptime:** %.1fs\n", proc.GetUptimeSeconds())

		// If the process is done and has a response, include it.
		if resp := proc.GetResponse(); resp != nil {
			b.WriteString("\n### Response\n\n")
			b.WriteString(formatChatResponse(resp))
		}

		return helpers.TextResult(b.String()), nil
	}
}

// --- list_active ---

// ListActiveSchema returns the JSON Schema for the list_active tool.
func ListActiveSchema() *structpb.Struct {
	s, _ := structpb.NewStruct(map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	})
	return s
}

// ListActive returns a tool handler that lists all tracked Claude Code CLI
// sessions with their current status.
func ListActive(bridge *Bridge) plugin.ToolHandler {
	return func(ctx context.Context, req *pluginv1.ToolRequest) (*pluginv1.ToolResponse, error) {
		procs := bridge.Plugin.ListProcesses()

		if len(procs) == 0 {
			return helpers.TextResult("## Active Sessions\n\nNo active sessions.\n"), nil
		}

		var b strings.Builder
		fmt.Fprintf(&b, "## Active Sessions (%d)\n\n", len(procs))
		fmt.Fprintf(&b, "| Session ID | Status | PID | Uptime |\n")
		fmt.Fprintf(&b, "|------------|--------|-----|--------|\n")

		for _, proc := range procs {
			status := "finished"
			if proc.IsRunning() {
				status = "running"
			}
			fmt.Fprintf(&b, "| %s | %s | %d | %.1fs |\n",
				proc.GetSessionID(), status, proc.GetPID(), proc.GetUptimeSeconds())
		}

		return helpers.TextResult(b.String()), nil
	}
}
