package bridgeclaude

import (
	"context"
	"log"

	"github.com/orchestra-mcp/plugin-bridge-claude/internal"
	"github.com/orchestra-mcp/sdk-go/plugin"
)

// Cleanup is a function that should be called during shutdown.
type Cleanup func()

// Register adds all Claude bridge tools to the builder and starts the
// PreToolUse permission HTTP server. Returns a cleanup function that kills
// all running claude processes and stops the permission server.
func Register(builder *plugin.PluginBuilder) Cleanup {
	return RegisterWithContext(context.Background(), builder)
}

// RegisterWithContext is like Register but accepts a context for the permission
// server lifecycle. When ctx is cancelled, the permission server stops.
func RegisterWithContext(ctx context.Context, builder *plugin.PluginBuilder) Cleanup {
	bp := internal.NewBridgePlugin()
	bp.RegisterTools(builder)
	if err := bp.StartPermissionServer(ctx); err != nil {
		log.Printf("[bridge.claude] permission server unavailable: %v", err)
	}
	return bp.KillAll
}
