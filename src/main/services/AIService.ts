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
    // Clean up previous provider
    if (this.currentProvider !== 'openai') {
      this.cleanup();
    }
    
    this.openai = createOpenAI({ apiKey });
    this.currentProvider = 'openai';
    this.currentModel = model;
    console.log('[AIService] Initialized with OpenAI model:', model);
  }

  /**
   * Initialize with Google Gemini API key
   */
  initializeGemini(apiKey: string, model: string = 'gemini-2.5-flash-lite'): void {
    // Clean up previous provider
    if (this.currentProvider !== 'gemini') {
      this.cleanup();
    }
    
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
    
    // Clean up previous provider (stop online services)
    if (this.currentProvider !== 'offline') {
      this.cleanup();
    }
    
    const result = await llamaService.startServer(modelPath);
    
    if (result.success) {
      this.currentProvider = 'offline';
      // Set the current model based on what was passed in
      if (modelPath) {
        // If it's a model ID (no path separators)
        if (!modelPath.includes('/') && !modelPath.includes('\\')) {
          this.currentModel = modelPath;
        } else {
          // Extract model name from path
          this.currentModel = modelPath.includes('functiongemma') 
            ? 'functiongemma-270m-q4' 
            : 'llama3.2-3b-q4';
        }
      } else {
        this.currentModel = 'llama3.2-3b-q4'; // default
      }
      console.log('[AIService] Offline model initialized successfully:', this.currentModel);
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
   * Clean up current provider (stop servers, clear instances)
   */
  cleanup(): void {
    console.log(`[AIService] Cleaning up current provider: ${this.currentProvider}`);
    
    // Clear API instances
    this.openai = null;
    this.google = null;
    
    // If switching away from offline, stop the llama server
    if (this.currentProvider === 'offline') {
      llamaService.stopServer();
      console.log('[AIService] Stopped offline llama server');
    }
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
   * Detect if user message requires MCP tool usage
   * Only returns true for VERY EXPLICIT action requests
   */
  private shouldUseMCPTools(message: string): boolean {
    const lower = message.toLowerCase().trim();
    
    // Very explicit patterns that clearly request actions
    const actionPatterns = [
      /^(list|show|get|find|display)\s+(all\s+)?(staff|tasks?|calls?|active)/i,
      /^call\s+(staff|someone|[\w\s]+)(\s+named|\s+called)?\s+/i,
      /^(create|add|make)\s+(new\s+)?(staff|task)/i,
      /^(update|change|modify)\s+(staff|task)/i,
      /^assign\s+task/i,
      /^complete\s+task/i,
      /^get\s+(staff|task|call)\s+(profile|status|result)/i,
    ];
    
    // Check if message matches any action pattern
    const isActionRequest = actionPatterns.some(pattern => pattern.test(lower));
    
    console.log(`[AIService] Tool detection: "${message.substring(0, 50)}..." -> ${isActionRequest ? 'ACTION' : 'CHAT'}`);
    return isActionRequest;
  }

  /**
   * Chat with offline model (WITH selective MCP tool support)
   */
  private async chatOffline(userMessage: string): Promise<ChatResult> {
    const mcpService = getMCPService();
    const toolCallTracker: ToolCallInfo[] = [];

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

      // Detect if this message needs tools
      const needsTools = this.shouldUseMCPTools(userMessage);
      let toolsToPass: Array<{ type: string; function: { name: string; description: string; parameters: Record<string, unknown> } }> | undefined;

      if (needsTools) {
        // Get MCP tools and convert to OpenAI format
        const mcpTools = mcpService.getAllTools();
        if (mcpTools.length > 0) {
          toolsToPass = mcpTools.map((mcpTool: MCPTool) => ({
            type: 'function',
            function: {
              name: mcpTool.name,
              description: mcpTool.description,
              parameters: mcpTool.inputSchema,
            },
          }));
          console.log(`[AIService] 🔧 Passing ${toolsToPass.length} tools to offline model`);
        }
      } else {
        console.log(`[AIService] 💬 Conversational mode - no tools`);
      }

      // Get response from local llama server
      const result = await llamaService.chat(messages, toolsToPass);

      // Handle tool calls if present
      if (result.toolCalls && result.toolCalls.length > 0) {
        console.log(`[AIService] ⚡ Model requested ${result.toolCalls.length} tool call(s)`);
        
        // Execute all tool calls
        for (const toolCall of result.toolCalls) {
          const toolName = toolCall.function.name;
          let toolArgs: Record<string, unknown>;
          
          try {
            toolArgs = JSON.parse(toolCall.function.arguments);
          } catch (parseError) {
            console.error('[AIService] Failed to parse tool arguments:', toolCall.function.arguments);
            continue;
          }
          
          const callInfo: ToolCallInfo = {
            id: toolCall.id,
            name: toolName,
            arguments: JSON.stringify(toolArgs, null, 2),
            status: 'pending',
          };
          toolCallTracker.push(callInfo);

          try {
            console.log(`[AIService] Executing: ${toolName}`, toolArgs);
            const toolResult = await mcpService.executeTool(toolName, toolArgs);
            
            if (toolResult.success) {
              const resultStr = typeof toolResult.result === 'string'
                ? toolResult.result
                : JSON.stringify(toolResult.result, null, 2);
              callInfo.result = resultStr;
              callInfo.status = 'success';
              console.log(`[AIService] ✓ Tool ${toolName} succeeded`);
            } else {
              callInfo.result = toolResult.error || 'Unknown error';
              callInfo.status = 'error';
              console.error(`[AIService] ✗ Tool ${toolName} failed:`, toolResult.error);
            }
          } catch (error) {
            callInfo.result = error instanceof Error ? error.message : 'Unknown error';
            callInfo.status = 'error';
            console.error(`[AIService] ✗ Tool ${toolName} threw error:`, error);
          }
        }

        // Add tool call to history
        this.conversationHistory.push({
          role: 'assistant',
          content: result.content || `Executed ${toolCallTracker.length} tool call(s)`,
        });

        // Return with tool calls
        return {
          response: result.content || 'Tool execution complete',
          toolCalls: toolCallTracker,
        };
      }

      // No tool calls - regular text response
      const response = result.content || '';

      // Add assistant response to history
      if (response) {
        this.conversationHistory.push({
          role: 'assistant',
          content: response,
        });
      }

      console.log(`[AIService] Response: "${response.substring(0, 100)}..."`);

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
   * Stream chat with offline model (WITH selective MCP tool support)
   */
  private async chatStreamOffline(userMessage: string, callbacks: StreamCallbacks): Promise<void> {
    const mcpService = getMCPService();

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

      // Detect if this message needs tools
      const needsTools = this.shouldUseMCPTools(userMessage);
      let toolsToPass: Array<{ type: string; function: { name: string; description: string; parameters: Record<string, unknown> } }> | undefined;

      if (needsTools) {
        // Get MCP tools and convert to OpenAI format
        const mcpTools = mcpService.getAllTools();
        if (mcpTools.length > 0) {
          toolsToPass = mcpTools.map((mcpTool: MCPTool) => ({
            type: 'function',
            function: {
              name: mcpTool.name,
              description: mcpTool.description,
              parameters: mcpTool.inputSchema,
            },
          }));
          console.log(`[AIService] 🔧 Passing ${toolsToPass.length} tools to offline model (streaming)`);
        }
      } else {
        console.log(`[AIService] 💬 Conversational mode - no tools (streaming)`);
      }

      // Get response from local llama server
      const result = await llamaService.chat(messages, toolsToPass);

      // Handle tool calls if present
      if (result.toolCalls && result.toolCalls.length > 0) {
        console.log(`[AIService] ⚡ Model requested ${result.toolCalls.length} tool call(s) in stream`);
        
        // Execute all tool calls
        const toolResults: string[] = [];
        for (const toolCall of result.toolCalls) {
          const toolName = toolCall.function.name;
          let toolArgs: Record<string, unknown>;
          
          try {
            toolArgs = JSON.parse(toolCall.function.arguments);
          } catch (parseError) {
            console.error('[AIService] Failed to parse tool arguments:', toolCall.function.arguments);
            continue;
          }

          // Notify about the tool call (for card rendering)
          const callInfo: ToolCallInfo = {
            id: toolCall.id,
            name: toolName,
            arguments: JSON.stringify(toolArgs, null, 2),
            status: 'pending',
          };
          callbacks.onToolCall(callInfo);

          try {
            console.log(`[AIService] Executing: ${toolName}`, toolArgs);
            const toolResult = await mcpService.executeTool(toolName, toolArgs);
            
            if (toolResult.success) {
              const resultStr = typeof toolResult.result === 'string'
                ? toolResult.result
                : JSON.stringify(toolResult.result, null, 2);
              toolResults.push(`✓ ${toolName}: ${resultStr}`);
              callInfo.result = resultStr;
              callInfo.status = 'success';
              callbacks.onToolResult(toolCall.id, resultStr, 'success');
              console.log(`[AIService] ✓ Tool ${toolName} succeeded`);
            } else {
              const errorMsg = toolResult.error || 'Unknown error';
              toolResults.push(`✗ ${toolName}: ${errorMsg}`);
              callInfo.result = errorMsg;
              callInfo.status = 'error';
              callbacks.onToolResult(toolCall.id, errorMsg, 'error');
              console.error(`[AIService] ✗ Tool ${toolName} failed:`, toolResult.error);
            }
          } catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            toolResults.push(`✗ ${toolName}: ${errorMsg}`);
            callInfo.result = errorMsg;
            callInfo.status = 'error';
            callbacks.onToolResult(toolCall.id, errorMsg, 'error');
            console.error(`[AIService] ✗ Tool ${toolName} threw error:`, error);
          }
        }

        // Don't stream raw JSON text - cards will display the results
        // Only send a brief completion message
        const completionMessage = `Executed ${result.toolCalls.length} tool call(s)`;
        
        // Add to history (for context in future messages)
        this.conversationHistory.push({
          role: 'assistant',
          content: completionMessage,
        });

        console.log(`[AIService] Offline stream complete with ${result.toolCalls.length} tool call(s)`);
        callbacks.onComplete(completionMessage);
        return;
      }

      // No tool calls - regular text response
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
