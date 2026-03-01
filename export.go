package bridgeclaude

import (
	"github.com/orchestra-mcp/plugin-bridge-claude/internal"
	"github.com/orchestra-mcp/sdk-go/plugin"
)

// Cleanup is a function that should be called during shutdown.
type Cleanup func()

// Register adds all Claude bridge tools to the builder.
// Returns a cleanup function that kills all running claude processes.
func Register(builder *plugin.PluginBuilder) Cleanup {
	bp := internal.NewBridgePlugin()
	bp.RegisterTools(builder)
	return bp.KillAll
}
