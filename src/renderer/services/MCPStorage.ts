/**
 * MCPStorage - Persistent storage for MCP server configurations
 * Saves MCP server paths to localStorage for reconnection on app restart
 */

export interface SavedMCPServer {
  id: string;
  name: string;
  scriptPath: string;
  savedAt: number;
  lastConnected?: number;
  connectionCount: number;
}

const STORAGE_KEY = 'goated_mcp_servers';

/**
 * Get all saved MCP servers from localStorage
 */
export function getSavedServers(): SavedMCPServer[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    
    const servers = JSON.parse(stored) as SavedMCPServer[];
    return servers.sort((a, b) => (b.lastConnected || 0) - (a.lastConnected || 0));
  } catch (error) {
    console.error('[MCPStorage] Failed to load saved servers:', error);
    return [];
  }
}

/**
 * Save an MCP server configuration
 */
export function saveServer(scriptPath: string, name?: string): SavedMCPServer {
  const servers = getSavedServers();
  
  // Check if server already exists
  const existing = servers.find(s => s.scriptPath === scriptPath);
  if (existing) {
    // Update existing server
    existing.lastConnected = Date.now();
    existing.connectionCount += 1;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(servers));
    return existing;
  }
  
  // Create new server entry
  const newServer: SavedMCPServer = {
    id: `mcp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    name: name || scriptPath.split('/').pop()?.replace(/\.(py|js)$/, '') || 'Unknown Server',
    scriptPath,
    savedAt: Date.now(),
    lastConnected: Date.now(),
    connectionCount: 1,
  };
  
  servers.push(newServer);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(servers));
  
  console.log('[MCPStorage] Saved server:', newServer);
  return newServer;
}

/**
 * Remove a saved MCP server
 */
export function removeSavedServer(id: string): void {
  const servers = getSavedServers();
  const filtered = servers.filter(s => s.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
  console.log('[MCPStorage] Removed server:', id);
}

/**
 * Update last connected timestamp
 */
export function updateLastConnected(scriptPath: string): void {
  const servers = getSavedServers();
  const server = servers.find(s => s.scriptPath === scriptPath);
  if (server) {
    server.lastConnected = Date.now();
    server.connectionCount += 1;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(servers));
  }
}

/**
 * Clear all saved servers
 */
export function clearAllServers(): void {
  localStorage.removeItem(STORAGE_KEY);
  console.log('[MCPStorage] Cleared all saved servers');
}
