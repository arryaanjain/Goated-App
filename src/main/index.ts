import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'path';
import { config } from 'dotenv';
import { getWhisperService, WhisperModel } from './services/WhisperService';
import { getMCPService } from './services/MCPService';
import { getAIService } from './services/AIService';

// Load environment variables from .env file
config();

let mainWindow: BrowserWindow | null = null;

// Initialize services
const whisperService = getWhisperService();
const mcpService = getMCPService();
const aiService = getAIService();

const isDev = process.env.NODE_ENV !== 'production' || !app.isPackaged;

function createWindow(): void {
  // Create the browser window with security settings
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 1200,
    minHeight: 800,
    title: 'GoatedApp',
    webPreferences: {
      // Security: Enable context isolation
      contextIsolation: true,
      // Security: Disable node integration in renderer
      nodeIntegration: false,
      // Security: Disable remote module
      // @ts-ignore - enableRemoteModule may not be in types but is important for security
      enableRemoteModule: false,
      // Load preload script
      preload: path.join(__dirname, '../preload/index.js'),
      // Security: Disable web security only in dev mode
      webSecurity: !isDev,
    },
    // Show window when ready to prevent visual flash
    show: false,
    backgroundColor: '#FFFFFF',
  });

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
  });

  // Load the app
  if (isDev) {
    // In development, load from Vite dev server
    mainWindow.loadURL('http://localhost:5173');
    // Open DevTools in development
    mainWindow.webContents.openDevTools();
  } else {
    // In production, load the built files
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  }

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// This method will be called when Electron has finished initialization
app.whenReady().then(async () => {
  // Initialize WhisperService
  try {
    await whisperService.initialize();
    console.log('[GoatedApp] WhisperService initialized');
  } catch (error) {
    console.error('[GoatedApp] Failed to initialize WhisperService:', error);
  }

  // Try to load API keys from localStorage via a config file
  // Since we can't access localStorage from main process, we'll use a JSON file
  const userDataPath = app.getPath('userData');
  const configPath = path.join(userDataPath, 'api-config.json');
  
  try {
    const fs = await import('fs/promises');
    const configData = await fs.readFile(configPath, 'utf-8');
    const config = JSON.parse(configData);
    
    // Initialize with saved API keys
    if (config.openaiApiKey) {
      aiService.initializeOpenAI(config.openaiApiKey, config.selectedModel || 'gpt-4o-mini');
      console.log('[GoatedApp] AIService initialized with OpenAI from saved config');
    } else if (config.geminiApiKey) {
      aiService.initializeGemini(config.geminiApiKey, config.selectedModel || 'gemini-2.5-flash-lite');
      console.log('[GoatedApp] AIService initialized with Gemini from saved config');
    } else {
      console.warn('[GoatedApp] No API keys found in config');
    }
  } catch (error) {
    console.log('[GoatedApp] No saved API config found - user needs to configure in Settings');
  }

  // Fallback: Try environment variables
  if (!aiService.isInitialized()) {
    const openaiKey = process.env.OPENAI_API_KEY;
    const geminiKey = process.env.GEMINI_API_KEY;
    
    if (openaiKey) {
      aiService.initializeOpenAI(openaiKey);
      console.log('[GoatedApp] AIService initialized with OpenAI from environment');
    } else if (geminiKey) {
      aiService.initializeGemini(geminiKey);
      console.log('[GoatedApp] AIService initialized with Gemini from environment');
    } else {
      console.warn('[GoatedApp] No API keys found - user must configure in Settings');
    }
  }

  createWindow();

  // On macOS, re-create window when dock icon is clicked
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit when all windows are closed (except on macOS)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Handle app quit
app.on('will-quit', () => {
  // Clean up resources here (e.g., stop Python backend)
  console.log('[GoatedApp] Application is quitting...');
});

// ============================================
// IPC Handlers
// ============================================

// Example IPC handler for getting app info
ipcMain.handle('app:getInfo', () => {
  return {
    name: app.getName(),
    version: app.getVersion(),
    platform: process.platform,
    arch: process.arch,
    userDataPath: app.getPath('userData'),
  };
});

// Backend status - now returns AI service status instead of Python backend
ipcMain.handle('backend:status', async () => {
  const status = aiService.getStatus();
  return {
    running: status.initialized,
    pid: null,
    restartCount: 0,
    healthy: status.initialized,
    provider: status.provider,
    model: status.model,
  };
});

// Transcription handler - Requirement 4.10: IPC handler at 'transcription:transcribe'
ipcMain.handle('transcription:transcribe', async (_event, audioBuffer: ArrayBuffer) => {
  try {
    const result = await whisperService.transcribeBuffer(audioBuffer);
    return result;
  } catch (error) {
    console.error('[Transcription] Error:', error);
    throw error;
  }
});

// Get available Whisper models
ipcMain.handle('transcription:getModels', () => {
  return whisperService.getAvailableModels();
});

// Set Whisper model
ipcMain.handle('transcription:setModel', async (_event, modelName: string) => {
  try {
    await whisperService.setModel(modelName as WhisperModel);
    return { success: true };
  } catch (error) {
    console.error('[Transcription] Failed to set model:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
});

// Get current Whisper model
ipcMain.handle('transcription:getModel', () => {
  return whisperService.getModelName();
});

// Chat handler - uses AIService with MCP tools
ipcMain.handle('chat:send', async (_event, message: string) => {
  try {
    if (!aiService.isInitialized()) {
      throw new Error('AI Service not initialized. Please configure API keys in Settings.');
    }

    console.log('[Chat] Using AIService with Vercel AI SDK');
    const result = await aiService.chat(message);
    return {
      response: result.response,
      toolCalls: result.toolCalls?.map(tc => ({
        id: tc.id,
        name: tc.name,
        arguments: tc.arguments,
        result: tc.result,
        status: tc.status,
      })),
    };
  } catch (error) {
    console.error('[Chat] Error:', error);
    throw error;
  }
});

// Clear conversation history
ipcMain.handle('chat:clear', () => {
  aiService.clearHistory();
  return { success: true };
});

// Streaming chat handler
ipcMain.handle('chat:startStream', async (_event, message: string, streamId: string) => {
  try {
    if (!aiService.isInitialized()) {
      mainWindow?.webContents.send(`chat:stream:${streamId}`, {
        type: 'error',
        data: 'AI Service not initialized. Please set GEMINI_API_KEY.',
      });
      return;
    }

    console.log('[Chat] Starting stream with Vercel AI SDK');
    
    await aiService.chatStream(message, {
      onTextChunk: (chunk: string) => {
        mainWindow?.webContents.send(`chat:stream:${streamId}`, {
          type: 'text',
          data: chunk,
        });
      },
      onToolCall: (toolCall) => {
        mainWindow?.webContents.send(`chat:stream:${streamId}`, {
          type: 'tool-call',
          data: toolCall,
        });
      },
      onToolResult: (toolCallId: string, result: string, status: 'success' | 'error') => {
        mainWindow?.webContents.send(`chat:stream:${streamId}`, {
          type: 'tool-result',
          data: { toolCallId, result, status },
        });
      },
      onComplete: (fullText: string) => {
        mainWindow?.webContents.send(`chat:stream:${streamId}`, {
          type: 'complete',
          data: fullText,
        });
      },
      onError: (error: Error) => {
        mainWindow?.webContents.send(`chat:stream:${streamId}`, {
          type: 'error',
          data: error.message,
        });
      },
    });
  } catch (error) {
    console.error('[Chat] Stream error:', error);
    mainWindow?.webContents.send(`chat:stream:${streamId}`, {
      type: 'error',
      data: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

// Placeholder for provider operations
ipcMain.handle('provider:list', () => {
  return [
    {
      id: 'local',
      name: 'Local (FunctionGemma)',
      type: 'local',
      baseURL: 'http://127.0.0.1:8000/v1',
      isDefault: true,
    },
  ];
});

ipcMain.handle('provider:getActive', () => {
  return {
    id: 'local',
    name: 'Local (FunctionGemma)',
    type: 'local',
    baseURL: 'http://127.0.0.1:8000/v1',
    isDefault: true,
  };
});

ipcMain.handle('provider:setActive', async (_event, _providerId: string) => {
  // Will be implemented in Provider Manager task
  return { success: true };
});

// ============================================
// MCP Operations - Model Context Protocol
// ============================================

// Connect to an MCP server by script path
ipcMain.handle('mcp:connect', async (_event, scriptPath: string) => {
  console.log('[MCP] Connect request for:', scriptPath);
  try {
    const result = await mcpService.connect(scriptPath);
    if (result.success && result.server) {
      return {
        success: true,
        server: {
          id: result.server.id,
          name: result.server.name,
          scriptPath: result.server.scriptPath,
          status: result.server.status,
          toolCount: result.server.tools.length,
          tools: result.server.tools.map(t => ({
            name: t.name,
            description: t.description,
          })),
        },
      };
    }
    return { success: false, error: result.error };
  } catch (error) {
    console.error('[MCP] Connect error:', error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    };
  }
});

// Disconnect from an MCP server
ipcMain.handle('mcp:disconnect', async (_event, serverId: string) => {
  console.log('[MCP] Disconnect request for:', serverId);
  try {
    const result = await mcpService.disconnect(serverId);
    return result;
  } catch (error) {
    console.error('[MCP] Disconnect error:', error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    };
  }
});

// List all connected MCP servers
ipcMain.handle('mcp:listServers', () => {
  const servers = mcpService.getConnectedServers();
  return servers.map(s => ({
    id: s.id,
    name: s.name,
    scriptPath: s.scriptPath,
    status: s.status,
    toolCount: s.tools.length,
    tools: s.tools.map(t => ({
      name: t.name,
      description: t.description,
    })),
    connectedAt: s.connectedAt,
  }));
});

// List all available tools from connected servers
ipcMain.handle('mcp:listTools', () => {
  return mcpService.getAllTools();
});

// Execute a tool on a connected MCP server
ipcMain.handle('mcp:executeTool', async (_event, toolName: string, args: Record<string, unknown>) => {
  console.log('[MCP] Execute tool:', toolName, args);
  try {
    const result = await mcpService.executeTool(toolName, args);
    return result;
  } catch (error) {
    console.error('[MCP] Execute tool error:', error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    };
  }
});

// Placeholder for model operations
ipcMain.handle('model:list', () => {
  return [
    {
      id: 'functiongemma',
      name: 'FunctionGemma 270M',
      type: 'llm',
      size: 270000000,
      status: 'not_downloaded',
    },
    {
      id: 'whisper-base',
      name: 'Whisper Base English',
      type: 'whisper',
      size: 142000000,
      status: 'not_downloaded',
    },
  ];
});

ipcMain.handle('model:download', async (_event, _modelId: string) => {
  // Will be implemented in Model Manager task
  return { success: false, error: 'Model download not yet implemented' };
});

ipcMain.handle('model:getDownloadProgress', async (_event, _modelId: string) => {
  return { modelId: _modelId, bytesDownloaded: 0, totalBytes: 0, percentage: 0 };
});

// API Key management
ipcMain.handle('api:setKeys', async (_event, config: { openaiApiKey?: string; geminiApiKey?: string; selectedModel?: string; provider?: 'openai' | 'gemini' }) => {
  try {
    // Save config to file
    const userDataPath = app.getPath('userData');
    const configPath = path.join(userDataPath, 'api-config.json');
    const fs = await import('fs/promises');
    
    await fs.writeFile(configPath, JSON.stringify(config, null, 2), 'utf-8');
    console.log('[API] Config saved to:', configPath);
    
    // Initialize the appropriate service
    if (config.provider === 'openai' && config.openaiApiKey) {
      aiService.initializeOpenAI(config.openaiApiKey, config.selectedModel || 'gpt-4o-mini');
      console.log('[API] OpenAI API key updated');
    } else if (config.provider === 'gemini' && config.geminiApiKey) {
      aiService.initializeGemini(config.geminiApiKey, config.selectedModel || 'gemini-2.5-flash-lite');
      console.log('[API] Gemini API key updated');
    } else if (config.openaiApiKey) {
      // Default to OpenAI if no provider specified
      aiService.initializeOpenAI(config.openaiApiKey, config.selectedModel || 'gpt-4o-mini');
      console.log('[API] OpenAI API key updated (default)');
    } else if (config.geminiApiKey) {
      aiService.initializeGemini(config.geminiApiKey, config.selectedModel || 'gemini-2.5-flash-lite');
      console.log('[API] Gemini API key updated');
    }
    
    return { success: true };
  } catch (error) {
    console.error('[API] Failed to set API keys:', error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    };
  }
});

ipcMain.handle('api:getStatus', () => {
  const status = aiService.getStatus();
  return {
    initialized: status.initialized,
    provider: status.provider,
    currentModel: status.model,
  };
});

console.log('[GoatedApp] Main process initialized');
