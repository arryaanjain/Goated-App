/**
 * Preload Script for GoatedApp
 * 
 * This script runs in a privileged context and exposes a secure API
 * to the renderer process via contextBridge.
 * 
 * Security: Only expose specific, validated methods - never expose
 * the entire ipcRenderer or other Node.js APIs.
 */

import { contextBridge, ipcRenderer } from 'electron';

// Type definitions for the exposed API
export interface AppInfo {
  name: string;
  version: string;
  platform: string;
  arch: string;
  userDataPath: string;
}

export interface BackendStatus {
  running: boolean;
  pid: number | null;
  restartCount: number;
  healthy: boolean;
}

export interface Provider {
  id: string;
  name: string;
  type: 'local' | 'openai' | 'anthropic' | 'azure-openai';
  baseURL: string;
  apiKey?: string;
  isDefault?: boolean;
}

export interface MCPTool {
  name: string;
  description: string;
  inputSchema?: Record<string, unknown>;
  serverId?: string;
}

export interface MCPServerInfo {
  id: string;
  name: string;
  scriptPath: string;
  status: 'connecting' | 'connected' | 'disconnected' | 'error';
  toolCount: number;
  tools: Array<{ name: string; description: string }>;
  connectedAt?: number;
}

export interface MCPConnectResult {
  success: boolean;
  server?: MCPServerInfo;
  error?: string;
}

export interface MCPToolExecutionResult {
  success: boolean;
  result?: unknown;
  error?: string;
}

export interface MCPServerConfig {
  id: string;
  name: string;
  transport: 'stdio' | 'sse';
  command?: string;
  args?: string[];
  url?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  type: 'llm' | 'whisper';
  size: number;
  status: 'not_downloaded' | 'downloading' | 'downloaded';
  localPath?: string;
}

export interface DownloadProgress {
  modelId: string;
  bytesDownloaded: number;
  totalBytes: number;
  percentage: number;
}

export interface ChatResponse {
  response: string;
  toolCalls?: Array<{
    id: string;
    name: string;
    arguments: string;
    result?: string;
    status?: 'pending' | 'success' | 'error';
  }>;
}

export interface StreamToolCall {
  id: string;
  name: string;
  arguments: string;
  result?: string;
  status: 'pending' | 'success' | 'error';
}

export type StreamEventType = 'text' | 'tool-call' | 'tool-result' | 'complete' | 'error';

export interface StreamEvent {
  type: StreamEventType;
  data: string | StreamToolCall | { toolCallId: string; result: string; status: 'success' | 'error' };
}

// Define the API that will be exposed to the renderer
const api = {
  // App information
  app: {
    getInfo: (): Promise<AppInfo> => ipcRenderer.invoke('app:getInfo'),
  },

  // Backend (Python sidecar) operations
  backend: {
    getStatus: (): Promise<BackendStatus> => ipcRenderer.invoke('backend:status'),
    restart: (): Promise<void> => ipcRenderer.invoke('backend:restart'),
  },

  // Transcription operations
  transcription: {
    transcribe: (audioBuffer: ArrayBuffer): Promise<string> =>
      ipcRenderer.invoke('transcription:transcribe', audioBuffer),
    getModels: (): Promise<string[]> =>
      ipcRenderer.invoke('transcription:getModels'),
    setModel: (modelName: string): Promise<{ success: boolean; error?: string }> =>
      ipcRenderer.invoke('transcription:setModel', modelName),
    getModel: (): Promise<string> =>
      ipcRenderer.invoke('transcription:getModel'),
  },

  // Chat operations
  chat: {
    send: (message: string): Promise<ChatResponse> =>
      ipcRenderer.invoke('chat:send', message),
    // Start streaming chat - returns immediately, events come via callback
    startStream: (message: string, onEvent: (event: StreamEvent) => void): void => {
      // Generate unique stream ID
      const streamId = `stream_${Date.now()}`;
      
      // Set up event listener for this stream
      const handler = (_event: unknown, data: StreamEvent) => {
        onEvent(data);
        // Clean up on complete or error
        if (data.type === 'complete' || data.type === 'error') {
          ipcRenderer.removeListener(`chat:stream:${streamId}`, handler);
        }
      };
      
      ipcRenderer.on(`chat:stream:${streamId}`, handler);
      
      // Start the stream
      ipcRenderer.invoke('chat:startStream', message, streamId);
    },
    clear: (): Promise<{ success: boolean }> =>
      ipcRenderer.invoke('chat:clear'),
  },

  // Provider operations
  provider: {
    list: (): Promise<Provider[]> => ipcRenderer.invoke('provider:list'),
    getActive: (): Promise<Provider> => ipcRenderer.invoke('provider:getActive'),
    setActive: (providerId: string): Promise<{ success: boolean }> =>
      ipcRenderer.invoke('provider:setActive', providerId),
  },

  // MCP operations - Model Context Protocol
  mcp: {
    // Connect to an MCP server by script path (.py or .js)
    connect: (scriptPath: string): Promise<MCPConnectResult> =>
      ipcRenderer.invoke('mcp:connect', scriptPath),
    
    // Disconnect from a connected MCP server
    disconnect: (serverId: string): Promise<{ success: boolean; error?: string }> =>
      ipcRenderer.invoke('mcp:disconnect', serverId),
    
    // List all connected MCP servers
    listServers: (): Promise<MCPServerInfo[]> =>
      ipcRenderer.invoke('mcp:listServers'),
    
    // List all available tools from all connected servers
    listTools: (): Promise<MCPTool[]> => 
      ipcRenderer.invoke('mcp:listTools'),
    
    // Execute a tool on a connected MCP server
    executeTool: (toolName: string, args: Record<string, unknown>): Promise<MCPToolExecutionResult> =>
      ipcRenderer.invoke('mcp:executeTool', toolName, args),
  },

  // Model operations
  model: {
    list: (): Promise<ModelInfo[]> => ipcRenderer.invoke('model:list'),
    download: (modelId: string): Promise<{ success: boolean; error?: string }> =>
      ipcRenderer.invoke('model:download', modelId),
    getDownloadProgress: (modelId: string): Promise<DownloadProgress> =>
      ipcRenderer.invoke('model:getDownloadProgress', modelId),
  },

  // API key management
  api: {
    setKeys: (config: { 
      openaiApiKey?: string; 
      geminiApiKey?: string; 
      selectedModel?: string;
      provider?: 'openai' | 'gemini' | 'offline';
      offlineModelPath?: string;
    }): Promise<{ success: boolean; error?: string }> =>
      ipcRenderer.invoke('api:setKeys', config),
    getStatus: (): Promise<{ initialized: boolean; provider: string; currentModel: string }> =>
      ipcRenderer.invoke('api:getStatus'),
  },

  // Offline model operations
  offline: {
    startServer: (modelPath: string): Promise<{ success: boolean; error?: string }> =>
      ipcRenderer.invoke('offline:startServer', modelPath),
    stopServer: (): Promise<{ success: boolean; error?: string }> =>
      ipcRenderer.invoke('offline:stopServer'),
    getStatus: (): Promise<{ running: boolean; ready: boolean; modelPath: string | null }> =>
      ipcRenderer.invoke('offline:getStatus'),
  },
};

// Expose the API to the renderer process
contextBridge.exposeInMainWorld('api', api);

// Expose ipcRenderer for RxDB Electron plugin (Requirements: 1.1, 1.2)
// RxDB needs direct access to ipcRenderer for its IPC storage communication
contextBridge.exposeInMainWorld('electron', {
  ipcRenderer: {
    invoke: (channel: string, ...args: unknown[]) => ipcRenderer.invoke(channel, ...args),
    on: (channel: string, listener: (...args: unknown[]) => void) => {
      ipcRenderer.on(channel, (_event, ...args) => listener(...args));
    },
    removeListener: (channel: string, listener: (...args: unknown[]) => void) => {
      ipcRenderer.removeListener(channel, listener);
    },
  },
});

// Type declaration for the window object
declare global {
  interface Window {
    api: typeof api;
    electron: {
      ipcRenderer: {
        invoke: (channel: string, ...args: unknown[]) => Promise<unknown>;
        on: (channel: string, listener: (...args: unknown[]) => void) => void;
        removeListener: (channel: string, listener: (...args: unknown[]) => void) => void;
      };
    };
  }
}

console.log('[GoatedApp] Preload script loaded');
