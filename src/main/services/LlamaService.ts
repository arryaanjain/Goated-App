import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import fs from 'fs';

/**
 * LlamaService - Manages local llama.cpp server process
 * Handles spawning, lifecycle, and communication with llama-server
 */
export class LlamaService {
  private serverProcess: ChildProcess | null = null;
  private serverPort = 8080;
  private serverUrl = `http://localhost:${this.serverPort}`;
  private isReady = false;
  private modelPath: string | null = null;

  constructor() {
    console.log('[LlamaService] Initialized');
  }

  /**
   * Get the bundled model path
   */
  private getBundledModelPath(): string {
    // In production, model is in resources/models
    // In development, model is in project root/models
    const isDev = process.env.NODE_ENV !== 'production';
    
    if (isDev) {
      // Development: models in project root
      return path.join(process.cwd(), 'models', 'Llama-3.2-3B-Instruct-Q4_K_L.gguf');
    } else {
      // Production: models bundled in resources
      return path.join(process.resourcesPath, 'models', 'Llama-3.2-3B-Instruct-Q4_K_L.gguf');
    }
  }

  /**
   * Get the llama-server executable path
   */
  private getLlamaServerPath(): string {
    const isDev = process.env.NODE_ENV !== 'production';
    
    if (isDev) {
      // Development: bin in project root
      return path.join(process.cwd(), 'bin', 'llama-server');
    } else {
      // Production: llama-server bundled in resources
      return path.join(process.resourcesPath, 'bin', 'llama-server');
    }
  }

  /**
   * Start the llama-server process with bundled or custom model
   */
  async startServer(modelPath?: string): Promise<{ success: boolean; error?: string }> {
    try {
      // Use provided path or auto-detect bundled model
      const resolvedModelPath = modelPath || this.getBundledModelPath();
      
      // Validate model file exists
      if (!fs.existsSync(resolvedModelPath)) {
        throw new Error(`Model file not found: ${resolvedModelPath}`);
      }

      // Check if server is already running
      if (this.serverProcess && !this.serverProcess.killed) {
        console.log('[LlamaService] Server already running');
        return { success: true };
      }

      this.modelPath = resolvedModelPath;

      // Path to llama-server executable
      const llamaServerPath = this.getLlamaServerPath();
      
      if (!fs.existsSync(llamaServerPath)) {
        throw new Error(`llama-server not found at: ${llamaServerPath}`);
      }

      console.log('[LlamaService] Starting llama-server...');
      console.log('[LlamaService] Model:', resolvedModelPath);
      console.log('[LlamaService] Server:', llamaServerPath);

      // Spawn llama-server with required flags
      this.serverProcess = spawn(llamaServerPath, [
        '--model', resolvedModelPath,
        '--ctx-size', '2048',
        '--threads', '4',
        '--batch-size', '256',
        '--port', this.serverPort.toString(),
        '--host', '127.0.0.1',
        '--n-gpu-layers', '0', // CPU only - change if you want GPU support
      ]);

      // Handle stdout
      this.serverProcess!.stdout?.on('data', (data) => {
        const output = data.toString();
        console.log('[LlamaServer]', output);
        
        // Check if server is ready
        if (output.includes('HTTP server listening') || output.includes('llama server listening')) {
          this.isReady = true;
          console.log('[LlamaService] Server is ready!');
        }
      });

      // Handle stderr
      this.serverProcess!.stderr?.on('data', (data) => {
        console.error('[LlamaServer Error]', data.toString());
      });

      // Handle process exit
      this.serverProcess!.on('exit', (code, signal) => {
        console.log(`[LlamaService] Server exited with code ${code}, signal ${signal}`);
        this.isReady = false;
        this.serverProcess = null;
      });

      // Wait for server to be ready (max 120 seconds for model loading)
      await this.waitForServer(120000);

      return { success: true };
    } catch (error) {
      console.error('[LlamaService] Failed to start server:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Wait for the server to be ready
   */
  private async waitForServer(timeoutMs: number): Promise<void> {
    const startTime = Date.now();
    let lastStatusCode: number | null = null;
    
    console.log('[LlamaService] Waiting for server to load model (this may take 1-2 minutes)...');
    
    while (Date.now() - startTime < timeoutMs) {
      if (this.isReady) {
        return;
      }
      
      // Try to ping the server
      try {
        const response = await fetch(`${this.serverUrl}/health`);
        
        // 200 = ready, 503 = still loading (expected during model load)
        if (response.status === 200) {
          this.isReady = true;
          console.log('[LlamaService] Server is ready and model is loaded!');
          return;
        } else if (response.status === 503) {
          // Server is alive but still loading the model
          if (lastStatusCode !== 503) {
            console.log('[LlamaService] Server responded with 503 - model is still loading...');
            lastStatusCode = 503;
          }
        } else {
          console.warn(`[LlamaService] Unexpected status code: ${response.status}`);
        }
      } catch (error) {
        // Server not ready yet, continue waiting
        // This is expected during initial startup
      }
      
      // Wait 1 second before next check (reduced frequency for long loading)
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    throw new Error(`Server startup timeout after ${timeoutMs / 1000} seconds`);
  }

  /**
   * Stop the llama-server process
   */
  stopServer(): { success: boolean; error?: string } {
    try {
      if (!this.serverProcess || this.serverProcess.killed) {
        console.log('[LlamaService] Server not running');
        return { success: true };
      }

      console.log('[LlamaService] Stopping server...');
      this.serverProcess.kill('SIGTERM');
      this.isReady = false;
      this.serverProcess = null;

      return { success: true };
    } catch (error) {
      console.error('[LlamaService] Failed to stop server:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Send a chat completion request to the local server
   */
  async chat(
    messages: Array<{ role: string; content: string }>,
    tools?: Array<{ type: string; function: { name: string; description: string; parameters: Record<string, unknown> } }>
  ): Promise<{ content?: string; toolCalls?: Array<{ id: string; type: string; function: { name: string; arguments: string } }> }> {
    if (!this.isReady) {
      throw new Error('Llama server is not ready');
    }

    try {
      // When NO tools are provided, force conversational mode with a system message
      let finalMessages = messages;
      if (!tools || tools.length === 0) {
        // Prepend a system message to override the model's default function-calling template
        finalMessages = [
          {
            role: 'system',
            content: 'You are a helpful AI assistant. Respond naturally in plain conversational text. Do not use JSON format or function calls.',
          },
          ...messages,
        ];
        console.log(`[LlamaService] Conversational mode - no tools, added system override`);
      } else {
        console.log(`[LlamaService] Function-calling mode - ${tools.length} tools available`);
      }

      const requestBody: Record<string, unknown> = {
        messages: finalMessages,
        temperature: 0.7,
        max_tokens: 2048,
        stream: false,
      };

      // Include tools if provided (enables function calling)
      if (tools && tools.length > 0) {
        requestBody.tools = tools;
      }

      const response = await fetch(`${this.serverUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json() as {
        choices: Array<{ 
          message: { 
            content?: string;
            tool_calls?: Array<{ id: string; type: string; function: { name: string; arguments: string } }>;
          } 
        }>;
      };
      
      const message = data.choices[0].message;
      
      // Return both content and tool calls (if any)
      return {
        content: message.content,
        toolCalls: message.tool_calls,
      };
    } catch (error) {
      console.error('[LlamaService] Chat request failed:', error);
      throw error;
    }
  }

  /**
   * Get server status
   */
  getStatus(): { running: boolean; ready: boolean; modelPath: string | null } {
    return {
      running: this.serverProcess !== null && !this.serverProcess.killed,
      ready: this.isReady,
      modelPath: this.modelPath,
    };
  }

  /**
   * Get server URL
   */
  getServerUrl(): string {
    return this.serverUrl;
  }

  /**
   * Cleanup on app exit
   */
  cleanup(): void {
    if (this.serverProcess && !this.serverProcess.killed) {
      console.log('[LlamaService] Cleaning up...');
      this.serverProcess.kill('SIGTERM');
    }
  }
}

// Singleton instance
export const llamaService = new LlamaService();
