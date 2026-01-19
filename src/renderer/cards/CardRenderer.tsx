/**
 * CardRenderer Component
 * Orchestrates card selection and rendering based on tool output
 */

import React from 'react';
import { CardRendererProps, MCPResponse } from './types';
import { getCardForTool } from './index';
import { ToolCallCard } from '../components/ToolCallCard';

/**
 * Parse tool output JSON string
 * @param toolOutput - JSON string from MCP tool
 * @returns Parsed data or null if invalid
 */
function parseToolOutput(toolOutput: string): MCPResponse | null {
  if (!toolOutput || toolOutput.trim() === '') {
    return null;
  }

  try {
    const parsed = JSON.parse(toolOutput);
    return parsed;
  } catch (error) {
    console.error('[CardRenderer] Failed to parse tool output:', error);
    return null;
  }
}

/**
 * CardRenderer - Selects and renders the appropriate card for a tool
 */
export const CardRenderer: React.FC<CardRendererProps> = ({ 
  toolName, 
  toolOutput, 
  status 
}) => {
  // Parse the tool output
  const parsedOutput = parseToolOutput(toolOutput);

  // If parsing failed, show fallback
  if (!parsedOutput) {
    console.log('[CardRenderer] No parsed output, using fallback for:', toolName);
    return (
      <ToolCallCard
        toolCall={{
          id: `fallback-${Date.now()}`,
          name: toolName,
          arguments: '',
          result: toolOutput,
          status: status,
        }}
      />
    );
  }

  // Look up card component in registry
  const CardComponent = getCardForTool(toolName);

  // If no card found, use fallback
  if (!CardComponent) {
    console.log('[CardRenderer] No card registered for tool:', toolName);
    return (
      <ToolCallCard
        toolCall={{
          id: `fallback-${Date.now()}`,
          name: toolName,
          arguments: '',
          result: toolOutput,
          status: status,
        }}
      />
    );
  }

  // If tool failed, show error in fallback
  if (!parsedOutput.success) {
    console.log('[CardRenderer] Tool failed:', toolName, parsedOutput.error);
    return (
      <ToolCallCard
        toolCall={{
          id: `error-${Date.now()}`,
          name: toolName,
          arguments: '',
          result: parsedOutput.error || 'Tool execution failed',
          status: 'error',
        }}
      />
    );
  }

  // Render the custom card
  console.log('[CardRenderer] Rendering card for:', toolName, parsedOutput.data);
  return (
    <CardComponent
      data={parsedOutput.data}
      status={status}
    />
  );
};

export default CardRenderer;
