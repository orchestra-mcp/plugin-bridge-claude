package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	pluginv1 "github.com/orchestra-mcp/gen-go/orchestra/plugin/v1"
	"github.com/orchestra-mcp/sdk-go/helpers"
	"github.com/orchestra-mcp/sdk-go/plugin"
	"google.golang.org/protobuf/types/known/structpb"
)

// StdioPermissionRequest mirrors internal.StdioPermissionRequest for use in
// the tools package (avoids circular import).
type StdioPermissionRequest struct {
	RequestID string          `json:"requestId"`
	ToolName  string          `json:"toolName"`
	ToolInput json.RawMessage `json:"toolInput"`
	Reason    string          `json:"reason"`
	ToolUseID string          `json:"toolUseId"`
}

// StdioQuestionRequest mirrors internal.QuestionRequest.
type StdioQuestionRequest struct {
	RequestID string          `json:"requestId"`
	ToolUseID string          `json:"toolUseId"`
	Questions json.RawMessage `json:"questions"`
	RawInput  json.RawMessage `json:"rawInput,omitempty"`
}

// PermissionBridge provides access to pending permission/question requests from
// active Claude processes. Implemented by BridgePlugin in plugin.go.
type PermissionBridge interface {
	// DrainPendingPermissions collects all pending permission requests from
	// active processes (non-blocking channel reads).
	DrainPendingPermissions() []StdioPermissionRequest
	// DrainPendingQuestions collects all pending question requests from
	// active processes (non-blocking channel reads).
	DrainPendingQuestions() []StdioQuestionRequest
	// RespondPermission sends a permission decision to the Claude process
	// that owns the given request ID. Returns false if not found.
	RespondPermission(requestID string, approved bool) bool
	// RespondQuestion sends a question answer to the Claude process that
	// owns the given request ID. Returns false if not found.
	RespondQuestion(requestID, answer string) bool
}

// PermissionStore holds drained permission/question requests so they survive
// across polling calls. The Swift UI polls get_pending_permission repeatedly;
// we drain from channels once and hold them here until responded to.
type PermissionStore struct {
	mu          sync.Mutex
	permissions map[string]StdioPermissionRequest // requestID → request
	questions   map[string]StdioQuestionRequest   // requestID → request
	bridge      PermissionBridge
}

// NewPermissionStore creates a new store that drains from the given bridge.
func NewPermissionStore(bridge PermissionBridge) *PermissionStore {
	return &PermissionStore{
		permissions: make(map[string]StdioPermissionRequest),
		questions:   make(map[string]StdioQuestionRequest),
		bridge:      bridge,
	}
}

// drain pulls any new requests from active process channels into the store.
func (ps *PermissionStore) drain() {
	for _, pr := range ps.bridge.DrainPendingPermissions() {
		ps.permissions[pr.RequestID] = pr
	}
	for _, qr := range ps.bridge.DrainPendingQuestions() {
		ps.questions[qr.RequestID] = qr
	}
}

// GetPendingPermissionSchema returns the JSON Schema for get_pending_permission.
func GetPendingPermissionSchema() *structpb.Struct {
	s, _ := structpb.NewStruct(map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	})
	return s
}

// GetPendingPermission returns a tool handler that lists pending permission and
// question requests waiting for user approval in the Swift UI.
func GetPendingPermission(store *PermissionStore) plugin.ToolHandler {
	return func(ctx context.Context, req *pluginv1.ToolRequest) (*pluginv1.ToolResponse, error) {
		store.mu.Lock()
		defer store.mu.Unlock()

		// Drain any new requests from active processes.
		store.drain()

		// Build combined list of permissions and questions.
		var results []map[string]any

		for _, pr := range store.permissions {
			results = append(results, map[string]any{
				"id":         pr.RequestID,
				"type":       "permission",
				"tool_name":  pr.ToolName,
				"tool_input": pr.ToolInput,
				"reason":     pr.Reason,
			})
		}

		for _, qr := range store.questions {
			results = append(results, map[string]any{
				"id":        qr.RequestID,
				"type":      "question",
				"questions": qr.Questions,
			})
		}

		if len(results) == 0 {
			return helpers.TextResult("[]"), nil
		}

		data, err := json.Marshal(results)
		if err != nil {
			return helpers.ErrorResult("marshal_error", err.Error()), nil
		}
		return helpers.TextResult(string(data)), nil
	}
}

// RespondPermissionSchema returns the JSON Schema for respond_permission.
func RespondPermissionSchema() *structpb.Struct {
	s, _ := structpb.NewStruct(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"id": map[string]any{
				"type":        "string",
				"description": "Permission or question request ID from get_pending_permission",
			},
			"decision": map[string]any{
				"type":        "string",
				"description": "User decision: 'approve' or 'deny'",
				"enum":        []any{"approve", "deny"},
			},
			"answer": map[string]any{
				"type":        "string",
				"description": "Answer text for question requests (optional, used when type is 'question')",
			},
		},
		"required": []any{"id", "decision"},
	})
	return s
}

// RespondPermission returns a tool handler that delivers the user's decision
// for a pending permission or question request. This sends a control_response
// message back to the Claude process via stdin.
func RespondPermission(store *PermissionStore) plugin.ToolHandler {
	return func(ctx context.Context, req *pluginv1.ToolRequest) (*pluginv1.ToolResponse, error) {
		if err := helpers.ValidateRequired(req.Arguments, "id", "decision"); err != nil {
			return helpers.ErrorResult("validation_error", err.Error()), nil
		}

		id := helpers.GetString(req.Arguments, "id")
		decision := helpers.GetString(req.Arguments, "decision")
		answer := helpers.GetString(req.Arguments, "answer")

		if decision != "approve" && decision != "deny" {
			return helpers.ErrorResult("validation_error", "decision must be 'approve' or 'deny'"), nil
		}

		store.mu.Lock()

		// Check if it's a question request.
		if _, isQuestion := store.questions[id]; isQuestion {
			delete(store.questions, id)
			store.mu.Unlock()

			answerText := answer
			if answerText == "" {
				answerText = decision // fallback
			}

			ok := store.bridge.RespondQuestion(id, answerText)
			if !ok {
				return helpers.ErrorResult("not_found", fmt.Sprintf("question request %q not found in active processes", id)), nil
			}
			return helpers.TextResult(fmt.Sprintf("## Question Answered\n\nRequest `%s` answered: %s\n", id, answerText)), nil
		}

		// It's a permission request.
		delete(store.permissions, id)
		store.mu.Unlock()

		approved := decision == "approve"
		ok := store.bridge.RespondPermission(id, approved)
		if !ok {
			return helpers.ErrorResult("not_found", fmt.Sprintf("permission request %q not found in active processes", id)), nil
		}

		var b strings.Builder
		verb := "Approved"
		if !approved {
			verb = "Denied"
		}
		fmt.Fprintf(&b, "## Permission %s\n\n", verb)
		fmt.Fprintf(&b, "Request `%s` has been %sd.\n", id, strings.ToLower(verb[:len(verb)-1])+"e")
		return helpers.TextResult(b.String()), nil
	}
}
