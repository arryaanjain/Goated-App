/**
 * AIService - AI Chat Service using Vercel AI SDK
 * 
 * Provides chat functionality with OpenAI and Gemini with automatic tool execution
 * via connected MCP servers. Supports both streaming and non-streaming modes.
 */

import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createOpenAI } from '@ai-sdk/openai';
import { generateText, streamText, stepCountIs } from 'ai';
import { tool } from '@ai-sdk/provider-utils';
import { z } from 'zod';
import { getMCPService, MCPTool } from './MCPService';
import { llamaService } from './LlamaService';

// Types
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  toolCallId?: string;
  toolCalls?: ToolCallInfo[];
}

export interface ToolCallInfo {
  id: string;
  name: string;
  arguments: string;
  result?: string;
  status: 'pending' | 'success' | 'error';
}

export interface ChatResult {
  response: string;
  toolCalls?: ToolCallInfo[];
}

export interface StreamCallbacks {
  onTextChunk: (chunk: string) => void;
  onToolCall: (toolCall: ToolCallInfo) => void;
  onToolResult: (toolCallId: string, result: string, status: 'success' | 'error') => void;
  onComplete: (fullText: string) => void;
  onError: (error: Error) => void;
}

export type AIProvider = 'openai' | 'gemini' | 'offline';

/**
 * AIService class for managing AI chat with tool execution
 */
class AIService {
  private static instance: AIService;
  private openai: ReturnType<typeof createOpenAI> | null = null;
  private google: ReturnType<typeof createGoogleGenerativeAI> | null = null;
  private currentProvider: AIProvider = 'offline';
  private currentModel: string = 'llama3.2-3b-q4';
  private conversationHistory: Array<{ role: 'user' | 'assistant'; content: string }> = [];

  private constructor() {}

  static getInstance(): AIService {
    if (!AIService.instance) {
      AIService.instance = new AIService();
    }
    return AIService.instance;
  }

  /**
   * Initialize with OpenAI API key
   */
  initializeOpenAI(apiKey: string, model: string = 'gpt-4o-mini'): void {
    this.openai = createOpenAI({ apiKey });
    this.currentProvider = 'openai';
    this.currentModel = model;
    console.log('[AIService] Initialized with OpenAI model:', model);
  }

  /**
   * Initialize with Google Gemini API key
   */
  initializeGemini(apiKey: string, model: string = 'gemini-2.5-flash-lite'): void {
    this.google = createGoogleGenerativeAI({ apiKey });
    this.currentProvider = 'gemini';
    this.currentModel = model;
    console.log('[AIService] Initialized with Gemini model:', model);
  }

  /**
   * Initialize offline model with llama.cpp
   */
  async initializeOffline(modelPath?: string): Promise<{ success: boolean; error?: string }> {
    console.log('[AIService] Initializing offline model:', modelPath || 'auto-detect');
    
    const result = await llamaService.startServer(modelPath);
    
    if (result.success) {
      this.currentProvider = 'offline';
      this.currentModel = 'llama3.2-3b-q4';
      console.log('[AIService] Offline model initialized successfully');
    } else {
      console.error('[AIService] Failed to initialize offline model:', result.error);
    }
    
    return result;
  }

  /**
   * Set the active model
   */
  setModel(model: string): void {
    this.currentModel = model;
    console.log('[AIService] Model changed to:', model);
  }

  /**
   * Get current provider and model
   */
  getStatus(): { provider: AIProvider; model: string; initialized: boolean } {
    return {
      provider: this.currentProvider,
      model: this.currentModel,
      initialized: this.isInitialized(),
    };
  }

  /**
   * Check if the service is initialized
   */
  isInitialized(): boolean {
    if (this.currentProvider === 'offline') {
      const status = llamaService.getStatus();
      return status.running && status.ready;
    }
    return this.openai !== null || this.google !== null;
  }

  /**
   * Get the current model instance
   */
  private getModel(): any {
    if (this.currentProvider === 'openai' && this.openai) {
      return this.openai(this.currentModel);
    }
    if (this.currentProvider === 'gemini' && this.google) {
      return this.google(this.currentModel);
    }
    throw new Error('AI Service not initialized');
  }

  /**
   * Convert MCP tools to Vercel AI SDK tool format
   */
  private convertMCPToolsToAITools(mcpTools: MCPTool[], toolCallTracker: ToolCallInfo[]): Record<string, any> {
    const tools: Record<string, any> = {};
    const mcpService = getMCPService();

    for (const mcpTool of mcpTools) {
      tools[mcpTool.name] = tool({
        description: mcpTool.description,
        inputSchema: z.object(this.convertJsonSchemaToZod(mcpTool.inputSchema)),
        execute: async (args: Record<string, unknown>) => {
          console.log(`[AIService] Executing tool: ${mcpTool.name}`, args);
          
          // Track this tool call
          const callId = `call_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
          const toolCallInfo: ToolCallInfo = {
            id: callId,
            name: mcpTool.name,
            arguments: JSON.stringify(args, null, 2),
            status: 'pending',
          };
          toolCallTracker.push(toolCallInfo);
          
          try {
            const result = await mcpService.executeTool(mcpTool.name, args);
            if (result.success) {
              const resultStr = typeof result.result === 'string' 
                ? result.result 
                : JSON.stringify(result.result, null, 2);
              toolCallInfo.result = resultStr;
              toolCallInfo.status = 'success';
              console.log(`[AIService] Tool ${mcpTool.name} result:`, resultStr);
              return result.result;
            }
            toolCallInfo.result = result.error || 'Tool execution failed';
            toolCallInfo.status = 'error';
            throw new Error(result.error || 'Tool execution failed');
          } catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            toolCallInfo.result = errorMsg;
            toolCallInfo.status = 'error';
            throw error;
          }
        },
      });
    }

    return tools;
  }

  /**
   * Convert JSON Schema to Zod schema (simplified conversion)
   */
  private convertJsonSchemaToZod(schema: Record<string, unknown>): Record<string, z.ZodTypeAny> {
    const zodSchema: Record<string, z.ZodTypeAny> = {};
    
    const properties = schema.properties as Record<string, { type?: string; description?: string }> | undefined;
    const required = schema.required as string[] | undefined;

    if (properties) {
      for (const [key, prop] of Object.entries(properties)) {
        let zodType: z.ZodTypeAny;
        
        switch (prop.type) {
          case 'string':
            zodType = z.string();
            break;
          case 'number':
          case 'integer':
            zodType = z.number();
            break;
          case 'boolean':
            zodType = z.boolean();
            break;
          case 'array':
            zodType = z.array(z.unknown());
            break;
          case 'object':
            zodType = z.record(z.string(), z.unknown());
            break;
          default:
            zodType = z.unknown();
        }

        if (prop.description) {
          zodType = zodType.describe(prop.description);
        }

        if (!required?.includes(key)) {
          zodType = zodType.optional();
        }

        zodSchema[key] = zodType;
      }
    }

    return zodSchema;
  }

  /**
   * Chat with offline model (tool support disabled for stable conversational responses)
   */
  private async chatOffline(userMessage: string): Promise<ChatResult> {
    // Add user message to history
    this.conversationHistory.push({
      role: 'user',
      content: userMessage,
    });

    try {
      // Prepare messages
      const messages: Array<{ role: string; content: string }> = this.conversationHistory.map(msg => ({
        role: msg.role,
        content: msg.content,
      }));

      console.log(`[AIService] Offline chat - conversational mode (no tools)`);

      // Get response from local llama server WITHOUT tools for stable conversational responses
      const result = await llamaService.chat(messages);

      // No tool calls - regular text response
      const response = result.content || '';

      // Add assistant response to history
      if (response) {
        this.conversationHistory.push({
          role: 'assistant',
          content: response,
        });
      }

      console.log(`[AIService] Offline response: "${response.substring(0, 100)}..."`);

      return {
        response,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('[AIService] Offline chat error:', errorMessage);
      throw error;
    }
  }

  /**
   * Stream chat with offline model (simulates streaming by chunking the response)
   */
  private async chatStreamOffline(userMessage: string, callbacks: StreamCallbacks): Promise<void> {
    // Add user message to history
    this.conversationHistory.push({
      role: 'user',
      content: userMessage,
    });

    try {
      // Prepare messages for llama server
      const messages = this.conversationHistory.map(msg => ({
        role: msg.role,
        content: msg.content,
      }));

      // Get response from local llama server (without tools for streaming to keep it simple)
      const result = await llamaService.chat(messages);
      const responseText = result.content || '';

      // Simulate streaming by sending chunks
      const chunkSize = 5; // characters per chunk
      for (let i = 0; i < responseText.length; i += chunkSize) {
        const chunk = responseText.slice(i, i + chunkSize);
        callbacks.onTextChunk(chunk);
        // Small delay to simulate streaming
        await new Promise(resolve => setTimeout(resolve, 20));
      }

      // Add assistant response to history
      if (responseText) {
        this.conversationHistory.push({
          role: 'assistant',
          content: responseText,
        });
      }

      console.log(`[AIService] Offline stream complete: "${responseText.substring(0, 100)}..."`);
      callbacks.onComplete(responseText);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('[AIService] Offline stream error:', errorMessage);
      callbacks.onError(error instanceof Error ? error : new Error(errorMessage));
    }
  }

  /**
   * Send a chat message and get a response with tool execution
   */
  async chat(userMessage: string): Promise<ChatResult> {
    if (!this.isInitialized()) {
      throw new Error('AI Service not initialized. Please configure API keys in Settings.');
    }

    // For offline models, use simple chat without tool execution for now
    if (this.currentProvider === 'offline') {
      return this.chatOffline(userMessage);
    }

    const mcpService = getMCPService();
    const mcpTools = mcpService.getAllTools();
    const toolCalls: ToolCallInfo[] = [];

    // Add user message to history
    this.conversationHistory.push({
      role: 'user',
      content: userMessage,
    });

    try {
      // Build tools object if we have MCP tools - pass tracker for results
      const tools = mcpTools.length > 0 
        ? this.convertMCPToolsToAITools(mcpTools, toolCalls) 
        : undefined;

      console.log(`[AIService] Sending message with ${mcpTools.length} tools available`);
      console.log(`[AIService] Tools:`, mcpTools.map(t => t.name));

      // Generate response with automatic tool execution
      const result = await generateText({
        model: this.getModel(),
        messages: this.conversationHistory,
        tools,
        stopWhen: stepCountIs(5), // Allow multiple tool execution steps
      });

      console.log(`[AIService] Raw result text: "${result.text?.substring(0, 200)}..."`);
      console.log(`[AIService] Tool calls tracked: ${toolCalls.length}`);

      // Add assistant response to history
      const responseText = result.text || '';
      if (responseText) {
        this.conversationHistory.push({
          role: 'assistant',
          content: responseText,
        });
      }

      console.log(`[AIService] Response: "${responseText.substring(0, 100)}..." with ${toolCalls.length} tool calls`);

      return {
        response: responseText,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('[AIService] Chat error:', errorMessage);
      console.error('[AIService] Full error:', error);
      
      // Mark any pending tool calls as errors
      for (const tc of toolCalls) {
        if (tc.status === 'pending') {
          tc.status = 'error';
          tc.result = errorMessage;
        }
      }

      throw error;
    }
  }

  /**
   * Clear conversation history
   */
  clearHistory(): void {
    this.conversationHistory = [];
    console.log('[AIService] Conversation history cleared');
  }

  /**
   * Get current conversation history
   */
  getHistory(): Array<{ role: 'user' | 'assistant'; content: string }> {
    return [...this.conversationHistory];
  }

  /**
   * Stream a chat message with real-time text updates
   */
  async chatStream(userMessage: string, callbacks: StreamCallbacks): Promise<void> {
    if (!this.isInitialized()) {
      callbacks.onError(new Error('AI Service not initialized. Please configure API keys in Settings.'));
      return;
    }

    // For offline models, use non-streaming chat and simulate streaming
    if (this.currentProvider === 'offline') {
      return this.chatStreamOffline(userMessage, callbacks);
    }

    const mcpService = getMCPService();
    const mcpTools = mcpService.getAllTools();
    const toolCalls: ToolCallInfo[] = [];

    // Add user message to history
    this.conversationHistory.push({
      role: 'user',
      content: userMessage,
    });

    try {
      // Build tools with streaming callbacks
      const tools = mcpTools.length > 0 
        ? this.convertMCPToolsToAIToolsWithCallbacks(mcpTools, toolCalls, callbacks) 
        : undefined;

      console.log(`[AIService] Streaming message with ${mcpTools.length} tools available`);

      // Stream response with automatic tool execution
      const result = streamText({
        model: this.getModel(),
        messages: this.conversationHistory,
        tools,
        stopWhen: stepCountIs(5),
      });

      let fullText = '';

      // Process the stream
      for await (const chunk of result.textStream) {
        fullText += chunk;
        callbacks.onTextChunk(chunk);
      }

      // Wait for completion
      await result;
      
      // Add assistant response to history
      if (fullText) {
        this.conversationHistory.push({
          role: 'assistant',
          content: fullText,
        });
      }

      console.log(`[AIService] Stream complete: "${fullText.substring(0, 100)}..." with ${toolCalls.length} tool calls`);
      callbacks.onComplete(fullText);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('[AIService] Stream error:', errorMessage);
      
      // Mark any pending tool calls as errors
      for (const tc of toolCalls) {
        if (tc.status === 'pending') {
          tc.status = 'error';
          tc.result = errorMessage;
          callbacks.onToolResult(tc.id, errorMessage, 'error');
        }
      }

      callbacks.onError(error instanceof Error ? error : new Error(errorMessage));
    }
  }

  /**
   * Convert MCP tools to AI tools with streaming callbacks
   */
  private convertMCPToolsToAIToolsWithCallbacks(
    mcpTools: MCPTool[], 
    toolCallTracker: ToolCallInfo[],
    callbacks: StreamCallbacks
  ): Record<string, any> {
    const tools: Record<string, any> = {};
    const mcpService = getMCPService();

    for (const mcpTool of mcpTools) {
      tools[mcpTool.name] = tool({
        description: mcpTool.description,
        inputSchema: z.object(this.convertJsonSchemaToZod(mcpTool.inputSchema)),
        execute: async (args: Record<string, unknown>) => {
          console.log(`[AIService] Executing tool: ${mcpTool.name}`, args);
          
          const callId = `call_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
          const toolCallInfo: ToolCallInfo = {
            id: callId,
            name: mcpTool.name,
            arguments: JSON.stringify(args, null, 2),
            status: 'pending',
          };
          toolCallTracker.push(toolCallInfo);
          
          // Notify UI about the tool call
          callbacks.onToolCall(toolCallInfo);
          
          try {
            const result = await mcpService.executeTool(mcpTool.name, args);
            if (result.success) {
              const resultStr = typeof result.result === 'string' 
                ? result.result 
                : JSON.stringify(result.result, null, 2);
              toolCallInfo.result = resultStr;
              toolCallInfo.status = 'success';
              
              // Notify UI about the result
              callbacks.onToolResult(callId, resultStr, 'success');
              
              console.log(`[AIService] Tool ${mcpTool.name} result:`, resultStr);
              return result.result;
            }
            const errorMsg = result.error || 'Tool execution failed';
            toolCallInfo.result = errorMsg;
            toolCallInfo.status = 'error';
            callbacks.onToolResult(callId, errorMsg, 'error');
            throw new Error(errorMsg);
          } catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            toolCallInfo.result = errorMsg;
            toolCallInfo.status = 'error';
            callbacks.onToolResult(callId, errorMsg, 'error');
            throw error;
          }
        },
      });
    }

    return tools;
  }
}

// Export singleton instance getter
export function getAIService(): AIService {
  return AIService.getInstance();
}

export default AIService;
