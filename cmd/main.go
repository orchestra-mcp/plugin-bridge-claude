// Command bridge-claude is the entry point for the bridge.claude plugin
// binary. It spawns and controls Claude Code CLI processes. This plugin
// does NOT require storage -- it manages in-memory process state only.
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/orchestra-mcp/sdk-go/plugin"
	"github.com/orchestra-mcp/plugin-bridge-claude/internal"
)

func main() {
	builder := plugin.New("bridge.claude").
		Version("0.1.0").
		Description("Claude Code CLI bridge — spawns and controls claude processes").
		Author("Orchestra").
		Binary("bridge-claude").
		ProvidesAI("claude")

	bp := internal.NewBridgePlugin()
	bp.RegisterTools(builder)

	p := builder.BuildWithTools()
	p.ParseFlags()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		bp.KillAll() // kill all running claude processes + stop permission server
		cancel()
	}()

	// Start the PreToolUse permission HTTP server so the hook script can
	// forward tool-approval requests to the Swift UI.
	if err := bp.StartPermissionServer(ctx); err != nil {
		log.Printf("bridge.claude: permission server unavailable: %v", err)
		// Non-fatal — permissions will auto-approve if server is not running.
	}

	if err := p.Run(ctx); err != nil {
		log.Fatalf("bridge.claude: %v", err)
	}
}
