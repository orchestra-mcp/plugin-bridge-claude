package internal

import (
	"testing"
)

// TestExtractChunkText_ContentBlockDelta verifies that content_block_delta
// events yield the delta text.
func TestExtractChunkText_ContentBlockDelta(t *testing.T) {
	event := map[string]any{
		"type": "content_block_delta",
		"delta": map[string]any{
			"type": "text_delta",
			"text": "Hello ",
		},
	}

	got := extractChunkText(event)
	if got != "Hello " {
		t.Errorf("extractChunkText: got %q, want %q", got, "Hello ")
	}
}

// TestExtractChunkText_AssistantMessage verifies that assistant events with
// text content blocks yield the text.
func TestExtractChunkText_AssistantMessage(t *testing.T) {
	event := map[string]any{
		"type": "assistant",
		"message": map[string]any{
			"content": []any{
				map[string]any{"type": "text", "text": "World"},
			},
		},
	}

	got := extractChunkText(event)
	if got == "" {
		t.Error("extractChunkText: expected non-empty text for assistant event")
	}
}

// TestExtractChunkText_UnknownType verifies that unknown event types return
// an empty string (no false positives).
func TestExtractChunkText_UnknownType(t *testing.T) {
	event := map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"model": "claude-sonnet-4-6",
		},
	}

	got := extractChunkText(event)
	if got != "" {
		t.Errorf("extractChunkText: expected empty for message_start, got %q", got)
	}
}

// TestExtractChunkText_ResultEvent verifies that result events (final summary)
// are not yielded as chunks (to avoid double-counting).
func TestExtractChunkText_ResultEvent(t *testing.T) {
	event := map[string]any{
		"type":   "result",
		"result": "final answer",
	}

	got := extractChunkText(event)
	if got != "" {
		t.Errorf("extractChunkText: expected empty for result event, got %q", got)
	}
}

// TestExtractChunkText_ContentBlockDeltaMissingText verifies that a delta
// without a text field returns empty string.
func TestExtractChunkText_ContentBlockDeltaMissingText(t *testing.T) {
	event := map[string]any{
		"type":  "content_block_delta",
		"delta": map[string]any{"type": "input_json_delta", "partial_json": "{}"},
	}

	got := extractChunkText(event)
	if got != "" {
		t.Errorf("extractChunkText: expected empty for non-text delta, got %q", got)
	}
}
