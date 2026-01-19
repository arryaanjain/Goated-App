import { useEffect, useState, useRef, useCallback } from 'react';
import './styles/App.css';
import { Greeting } from './components/Greeting';
import { InputArea } from './components/InputArea';
import { Message } from './components/Message';
import { ToolCall } from './components/ToolCallCard';
import { SettingsModal } from './components/SettingsModal';
import { ConnectToolButton } from './components/ConnectToolButton';
import { ToolConnectionModal } from './components/ToolConnectionModal';
import { useRecorder } from './hooks/useRecorder';
import { useConversations } from './hooks/useConversations';
import { useMCP } from './hooks/useMCP';
import { ToolCallData } from './db/schemas';
import { StreamEvent, StreamToolCall } from '../preload/index';

interface AppInfo {
  name: string;
  version: string;
  platform: string;
  arch: string;
  userDataPath: string;
}

// UI message type with streaming state
interface MessageData {
  id: string;
  role: 'user' | 'assistant' | 'tool';
  content: string;
  timestamp: Date;
  toolCalls?: ToolCall[];
  isStreaming?: boolean;
}

function App() {
  const [appInfo, setAppInfo] = useState<AppInfo | null>(null);
  const [backendStatus, setBackendStatus] = useState<string>('Checking...');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [toolModalOpen, setToolModalOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [streamingContent, setStreamingContent] = useState<string>('');
  const [streamingToolCalls, setStreamingToolCalls] = useState<ToolCall[]>([]);
  const messageListRef = useRef<HTMLDivElement>(null);

  // Database-backed conversations hook
  const {
    conversations,
    activeConversationId,
    setActiveConversationId,
    messages: dbMessages,
    isLoading: isDbLoading,
    error: dbError,
    createConversation,
    deleteConversation,
    addMessage
  } = useConversations();

  // Audio recording hook
  const { isRecording, isTranscribing, error: recorderError, startRecording, stopRecording } = useRecorder();

  // MCP server management hook
  const {
    connectedServers,
    isConnecting,
    error: mcpError,
    connect: connectMCP,
    disconnect: disconnectMCP,
  } = useMCP();

  // Get user name for greeting (placeholder)
  const userName = appInfo?.name ? 'Doctor' : 'Doctor';

  // Convert DB messages to UI format, including streaming message
  const messages: MessageData[] = [
    ...dbMessages.map(msg => ({
      id: msg.id,
      role: msg.role,
      content: msg.content,
      timestamp: new Date(msg.timestamp),
      toolCalls: msg.toolCalls?.map(tc => ({
        id: tc.id,
        name: tc.name,
        arguments: tc.arguments,
        result: tc.result, // IMPORTANT: Include result for card rendering
        status: tc.status
      })),
    })),
    // Add streaming message if active
    ...(streamingMessageId ? [{
      id: streamingMessageId,
      role: 'assistant' as const,
      content: streamingContent,
      timestamp: new Date(),
      toolCalls: streamingToolCalls,
      isStreaming: true,
    }] : []),
  ];

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  }, [messages]);


  useEffect(() => {
    // Fetch app info on mount
    const fetchAppInfo = async () => {
      try {
        const info = await window.api.app.getInfo();
        setAppInfo(info);
      } catch (error) {
        console.error('Failed to get app info:', error);
      }
    };

    const fetchBackendStatus = async () => {
      try {
        const status = await window.api.backend.getStatus();
        if (status.running && status.healthy) {
          setBackendStatus('Online');
        } else if (status.running) {
          setBackendStatus('Starting...');
        } else {
          setBackendStatus('Offline');
        }
      } catch (error) {
        setBackendStatus('Error');
      }
    };

    fetchAppInfo();
    fetchBackendStatus();

    // Poll backend status every 5 seconds
    const interval = setInterval(fetchBackendStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Handle stream events
  const handleStreamEvent = useCallback((event: StreamEvent) => {
    switch (event.type) {
      case 'text':
        setStreamingContent(prev => prev + (event.data as string));
        break;
      case 'tool-call':
        const toolCall = event.data as StreamToolCall;
        setStreamingToolCalls(prev => [...prev, {
          id: toolCall.id,
          name: toolCall.name,
          arguments: toolCall.arguments,
          status: toolCall.status,
        }]);
        break;
      case 'tool-result':
        const result = event.data as { toolCallId: string; result: string; status: 'success' | 'error' };
        setStreamingToolCalls(prev => prev.map(tc => 
          tc.id === result.toolCallId 
            ? { ...tc, result: result.result, status: result.status }
            : tc
        ));
        break;
      case 'complete':
        // Will be handled in handleSubmit
        break;
      case 'error':
        console.error('[Stream] Error:', event.data);
        break;
    }
  }, []);

  const handleSubmit = async () => {
    if (!inputValue.trim() || isSending) return;
    
    const userContent = inputValue.trim();
    setInputValue('');
    setIsSending(true);

    try {
      // Add user message to database
      await addMessage('user', userContent);

      // Initialize streaming state
      const streamId = `stream-${Date.now()}`;
      setStreamingMessageId(streamId);
      setStreamingContent('');
      setStreamingToolCalls([]);

      // Start streaming chat
      await new Promise<void>((resolve, reject) => {
        let finalContent = '';
        let finalToolCalls: ToolCall[] = [];

        window.api.chat.startStream(userContent, (event: StreamEvent) => {
          handleStreamEvent(event);

          if (event.type === 'text') {
            finalContent += event.data as string;
          } else if (event.type === 'tool-call') {
            const tc = event.data as StreamToolCall;
            finalToolCalls.push({
              id: tc.id,
              name: tc.name,
              arguments: tc.arguments,
              status: tc.status,
            });
          } else if (event.type === 'tool-result') {
            const result = event.data as { toolCallId: string; result: string; status: 'success' | 'error' };
            finalToolCalls = finalToolCalls.map(tc =>
              tc.id === result.toolCallId
                ? { ...tc, result: result.result, status: result.status }
                : tc
            );
          } else if (event.type === 'complete') {
            // Save to database
            const toolCallData: ToolCallData[] | undefined = finalToolCalls.length > 0
              ? finalToolCalls.map(tc => ({
                  id: tc.id,
                  name: tc.name,
                  arguments: tc.arguments,
                  result: tc.result,
                  status: tc.status as 'pending' | 'success' | 'error',
                }))
              : undefined;

            addMessage('assistant', finalContent || 'No response received.', toolCallData)
              .then(() => {
                setStreamingMessageId(null);
                setStreamingContent('');
                setStreamingToolCalls([]);
                resolve();
              })
              .catch(reject);
          } else if (event.type === 'error') {
            reject(new Error(event.data as string));
          }
        });
      });

    } catch (error) {
      console.error('Chat error:', error);
      // Clear streaming state
      setStreamingMessageId(null);
      setStreamingContent('');
      setStreamingToolCalls([]);
      // Add error message to database
      await addMessage(
        'assistant',
        `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`
      );
    } finally {
      setIsSending(false);
    }
  };

  const handleMicClick = async () => {
    if (isRecording) {
      // Stop recording and get transcription
      try {
        const transcribedText = await stopRecording();
        if (transcribedText && transcribedText.trim()) {
          // Set the transcribed text in the input
          setInputValue(prev => prev ? `${prev} ${transcribedText}` : transcribedText);
        }
      } catch (error) {
        console.error('Failed to transcribe:', error);
      }
    } else {
      // Start recording
      try {
        await startRecording();
      } catch (error) {
        console.error('Failed to start recording:', error);
      }
    }
  };

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const handleNewSession = async () => {
    try {
      await window.api.chat.clear();
      await createConversation();
    } catch (error) {
      console.error('Failed to create new session:', error);
    }
  };

  const handleSelectConversation = (id: string) => {
    setActiveConversationId(id);
    setSidebarOpen(false);
  };

  const handleDeleteConversation = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteConversation(id);
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  };

  // Format timestamp for display
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    
    if (isToday) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };


  // Show loading state while database initializes
  if (isDbLoading) {
    return (
      <div className="app">
        <div className="loading-screen">
          <div className="loading-spinner" />
          <p>Loading conversations...</p>
        </div>
      </div>
    );
  }

  // Show error state if database failed
  if (dbError) {
    return (
      <div className="app">
        <div className="error-screen">
          <p>Failed to load conversations: {dbError}</p>
          <button onClick={() => window.location.reload()}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Sidebar Backdrop */}
      <div 
        className={`sidebar-backdrop ${sidebarOpen ? 'sidebar-backdrop--visible' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
        <div className="sidebar__drag-region" />
        <div className="sidebar-header">
          <button className="new-session-btn" onClick={handleNewSession}>
            <span>New Session</span>
            <span className="new-session-btn__icon">
              <span className="plus-icon">+</span>
            </span>
          </button>
        </div>
        <div className="sidebar-content">
          <div className="history-section__title">Conversations</div>
          <div className="conversation-list">
            {conversations.length > 0 ? (
              conversations.map(conv => (
                <div 
                  key={conv.id}
                  className={`conversation-item ${conv.id === activeConversationId ? 'conversation-item--active' : ''}`}
                  onClick={() => handleSelectConversation(conv.id)}
                >
                  <div className="conversation-item__content">
                    <span className="conversation-item__title">
                      {conv.preview || conv.title}
                    </span>
                    <span className="conversation-item__time">
                      {formatTime(conv.updatedAt)}
                    </span>
                  </div>
                  <button 
                    className="conversation-item__delete"
                    onClick={(e) => handleDeleteConversation(conv.id, e)}
                    aria-label="Delete conversation"
                  >
                    ×
                  </button>
                </div>
              ))
            ) : (
              <p className="empty-state">No conversations yet</p>
            )}
          </div>
        </div>
        <div className="sidebar-footer">
          <button 
            className="settings-btn"
            onClick={() => setSettingsOpen(true)}
            aria-label="Open settings"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
            </svg>
            <span>Settings</span>
          </button>
          <div className="user-profile">
            <div className="user-profile__avatar">DR</div>
            <span className="user-profile__name">{userName}</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {/* Hamburger Toggle for Mobile */}
        <button 
          className="hamburger-toggle"
          onClick={toggleSidebar}
          aria-label="Toggle sidebar"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>

        {/* Connect Tool Button */}
        <ConnectToolButton
          onClick={() => setToolModalOpen(true)}
          connectedCount={connectedServers.length}
        />

        {/* Chat Area */}
        <div className="chat-area">
          {messages.length === 0 ? (
            <Greeting 
              userName={userName}
              statusMessage={backendStatus === 'Online' ? 'System Secure & Online' : `System ${backendStatus}`}
            />
          ) : (
            <div className="message-list" ref={messageListRef}>
              {messages.map((message) => (
                <Message
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  timestamp={message.timestamp}
                  toolCalls={message.toolCalls}
                  isStreaming={message.isStreaming}
                />
              ))}
            </div>
          )}
        </div>

        {/* Input Area */}
        <InputArea
          isListening={isRecording || isTranscribing}
          value={inputValue}
          onChange={setInputValue}
          onSubmit={handleSubmit}
          onMicClick={handleMicClick}
          disabled={isSending || isTranscribing}
        />
        
        {/* Show recorder error if any */}
        {recorderError && (
          <div className="recorder-error">
            {recorderError}
          </div>
        )}
      </main>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />

      {/* Tool Connection Modal */}
      <ToolConnectionModal
        isOpen={toolModalOpen}
        onClose={() => setToolModalOpen(false)}
        connectedServers={connectedServers}
        onConnect={connectMCP}
        onDisconnect={disconnectMCP}
        isConnecting={isConnecting}
        error={mcpError}
      />
    </div>
  );
}

export default App;
