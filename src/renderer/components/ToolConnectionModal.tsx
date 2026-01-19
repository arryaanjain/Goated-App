import React, { useState, useEffect } from 'react';
import './ToolConnectionModal.css';
import { getSavedServers, saveServer, removeSavedServer, type SavedMCPServer } from '../services/MCPStorage';

interface MCPServerInfo {
  id: string;
  name: string;
  scriptPath: string;
  status: string;
  toolCount: number;
  tools: Array<{ name: string; description: string }>;
}

interface ToolConnectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  connectedServers: MCPServerInfo[];
  onConnect: (path: string) => Promise<void>;
  onDisconnect: (serverId: string) => Promise<void>;
  isConnecting: boolean;
  error: string | null;
}

/**
 * ToolConnectionModal Component - MCP Server Connection Interface
 * Modal for connecting to MCP servers and managing connections
 * Requirements: 2.1-2.8, 6.1-6.3
 */
export const ToolConnectionModal: React.FC<ToolConnectionModalProps> = ({
  isOpen,
  onClose,
  connectedServers,
  onConnect,
  onDisconnect,
  isConnecting,
  error,
}) => {
  const [serverPath, setServerPath] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);
  const [savedServers, setSavedServers] = useState<SavedMCPServer[]>([]);
  const [retryingServerId, setRetryingServerId] = useState<string | null>(null);

  // Load saved servers on mount
  useEffect(() => {
    if (isOpen) {
      setSavedServers(getSavedServers());
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const validatePath = (path: string): boolean => {
    const trimmed = path.trim();
    if (!trimmed) {
      setLocalError('Please enter a server script path');
      return false;
    }
    if (!trimmed.endsWith('.py') && !trimmed.endsWith('.js')) {
      setLocalError('Server script must be a .py or .js file');
      return false;
    }
    setLocalError(null);
    return true;
  };

  const handleConnect = async () => {
    if (!validatePath(serverPath)) return;
    
    try {
      await onConnect(serverPath.trim());
      // Save the server path on successful connection
      saveServer(serverPath.trim());
      setSavedServers(getSavedServers());
      setServerPath('');
      setLocalError(null);
    } catch (err) {
      // Error is handled by parent
    }
  };

  const handleRetry = async (server: SavedMCPServer) => {
    setRetryingServerId(server.id);
    try {
      await onConnect(server.scriptPath);
      // Update saved servers list
      setSavedServers(getSavedServers());
    } catch (err) {
      // Error is handled by parent
    } finally {
      setRetryingServerId(null);
    }
  };

  const handleRemoveSaved = (serverId: string) => {
    removeSavedServer(serverId);
    setSavedServers(getSavedServers());
  };

  const isServerConnected = (scriptPath: string): boolean => {
    return connectedServers.some(s => s.scriptPath === scriptPath);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isConnecting) {
      handleConnect();
    }
  };

  const displayError = localError || error;

  return (
    <div className="tool-modal__backdrop" onClick={handleBackdropClick}>
      <div 
        className="tool-modal" 
        role="dialog" 
        aria-modal="true" 
        aria-labelledby="tool-modal-title"
      >
        <div className="tool-modal__header">
          <h2 id="tool-modal-title" className="tool-modal__title">
            <svg 
              className="tool-modal__title-icon"
              width="24" 
              height="24" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2"
            >
              <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
            </svg>
            Connect MCP Server
          </h2>
          <button
            className="tool-modal__close-btn"
            onClick={onClose}
            aria-label="Close modal"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="tool-modal__content">
          {/* Connection Form */}
          <div className="tool-modal__section">
            <label className="tool-modal__label" htmlFor="server-path">
              Server Script Path
            </label>
            <p className="tool-modal__hint">
              Enter the path to a Python (.py) or Node.js (.js) MCP server script
            </p>
            <div className="tool-modal__input-group">
              <input
                id="server-path"
                type="text"
                className="tool-modal__input"
                placeholder="/path/to/server.py or /path/to/server.js"
                value={serverPath}
                onChange={(e) => setServerPath(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isConnecting}
              />
              <button
                className="tool-modal__connect-btn"
                onClick={handleConnect}
                disabled={isConnecting || !serverPath.trim()}
              >
                {isConnecting ? (
                  <>
                    <span className="tool-modal__spinner" />
                    Connecting...
                  </>
                ) : (
                  'Connect'
                )}
              </button>
            </div>
            {displayError && (
              <p className="tool-modal__error">{displayError}</p>
            )}
          </div>

          {/* Connected Servers List */}
          <div className="tool-modal__section">
            <h3 className="tool-modal__section-title">
              Connected Servers
              {connectedServers.length > 0 && (
                <span className="tool-modal__count">{connectedServers.length}</span>
              )}
            </h3>
            
            {connectedServers.length === 0 ? (
              <div className="tool-modal__empty">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                  <line x1="8" y1="21" x2="16" y2="21" />
                  <line x1="12" y1="17" x2="12" y2="21" />
                </svg>
                <p>No servers connected</p>
                <p className="tool-modal__empty-hint">
                  Connect an MCP server to enable AI tool capabilities
                </p>
              </div>
            ) : (
              <ul className="tool-modal__server-list">
                {connectedServers.map((server) => (
                  <li key={server.id} className="tool-modal__server-item tool-modal__server-item--connected">
                    <div className="tool-modal__server-info">
                      <div className="tool-modal__server-header">
                        <span className="tool-modal__server-status tool-modal__server-status--connected" />
                        <span className="tool-modal__server-name">{server.name}</span>
                        <span className="tool-modal__tool-count">
                          {server.toolCount} tool{server.toolCount !== 1 ? 's' : ''}
                        </span>
                      </div>
                      <p className="tool-modal__server-path">{server.scriptPath}</p>
                      {server.tools.length > 0 && (
                        <div className="tool-modal__tools-preview">
                          {server.tools.slice(0, 3).map((tool) => (
                            <span key={tool.name} className="tool-modal__tool-tag">
                              {tool.name}
                            </span>
                          ))}
                          {server.tools.length > 3 && (
                            <span className="tool-modal__tool-more">
                              +{server.tools.length - 3} more
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    <button
                      className="tool-modal__disconnect-btn"
                      onClick={() => onDisconnect(server.id)}
                      aria-label={`Disconnect ${server.name}`}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Saved Servers List */}
          {savedServers.length > 0 && (
            <div className="tool-modal__section">
              <h3 className="tool-modal__section-title">
                Saved Servers
                <span className="tool-modal__count">{savedServers.length}</span>
              </h3>
              <p className="tool-modal__hint tool-modal__hint--section">
                Previously connected servers - click retry to reconnect
              </p>
              
              <ul className="tool-modal__server-list">
                {savedServers.map((server) => {
                  const connected = isServerConnected(server.scriptPath);
                  const isRetrying = retryingServerId === server.id;
                  
                  return (
                    <li key={server.id} className={`tool-modal__server-item tool-modal__server-item--saved ${connected ? 'tool-modal__server-item--already-connected' : ''}`}>
                      <div className="tool-modal__server-info">
                        <div className="tool-modal__server-header">
                          <span className={`tool-modal__server-status ${connected ? 'tool-modal__server-status--connected' : 'tool-modal__server-status--saved'}`} />
                          <span className="tool-modal__server-name">{server.name}</span>
                          {connected && (
                            <span className="tool-modal__connected-badge">Connected</span>
                          )}
                        </div>
                        <p className="tool-modal__server-path">{server.scriptPath}</p>
                        <div className="tool-modal__server-meta">
                          <span className="tool-modal__server-meta-item">
                            Connected {server.connectionCount} time{server.connectionCount !== 1 ? 's' : ''}
                          </span>
                          {server.lastConnected && (
                            <span className="tool-modal__server-meta-item">
                              Last: {new Date(server.lastConnected).toLocaleDateString()}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="tool-modal__server-actions">
                        {!connected && (
                          <button
                            className="tool-modal__retry-btn"
                            onClick={() => handleRetry(server)}
                            disabled={isRetrying}
                            aria-label={`Retry connection to ${server.name}`}
                          >
                            {isRetrying ? (
                              <>
                                <span className="tool-modal__spinner tool-modal__spinner--small" />
                                Connecting...
                              </>
                            ) : (
                              <>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <polyline points="23 4 23 10 17 10" />
                                  <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
                                </svg>
                                Retry
                              </>
                            )}
                          </button>
                        )}
                        <button
                          className="tool-modal__remove-btn"
                          onClick={() => handleRemoveSaved(server.id)}
                          aria-label={`Remove ${server.name} from saved servers`}
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="3 6 5 6 21 6" />
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                          </svg>
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </div>

        <div className="tool-modal__footer">
          <button
            className="tool-modal__btn tool-modal__btn--secondary"
            onClick={onClose}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default ToolConnectionModal;
