package storage

import (
	"context"

	pluginv1 "github.com/orchestra-mcp/gen-go/orchestra/plugin/v1"
)

// StorageClient sends requests to the orchestrator for storage operations.
type StorageClient interface {
	Send(ctx context.Context, req *pluginv1.PluginRequest) (*pluginv1.PluginResponse, error)
}

// DataStorage wraps the storage client for tool handlers.
type DataStorage struct {
	client StorageClient
}

// NewDataStorage creates a new DataStorage with the given client.
func NewDataStorage(client StorageClient) *DataStorage {
	return &DataStorage{client: client}
}
