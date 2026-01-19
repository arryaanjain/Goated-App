import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './Message.css';
import { type ToolCall } from './ToolCallCard';
import { CardRenderer } from '../cards/CardRenderer';

interface MessageProps {
  role: 'user' | 'assistant' | 'tool';
  content: string;
  timestamp?: Date;
  toolCalls?: ToolCall[];
  isLoading?: boolean;
  isStreaming?: boolean;
}

/**
 * Message Component - Claude-style Message Bubble with Markdown Support
 * Displays messages with role-based styling and full markdown rendering
 */
export const Message: React.FC<MessageProps> = ({
  role,
  content,
  timestamp,
  toolCalls,
  isLoading,
  isStreaming,
}) => {
  // Memoize markdown components for performance
  const markdownComponents = useMemo(() => ({
    // Code blocks with syntax highlighting
    code: ({ node, inline, className, children, ...props }: any) => {
      const match = /language-(\w+)/.exec(className || '');
      const language = match ? match[1] : '';
      
      if (inline) {
        return (
          <code className="md-inline-code" {...props}>
            {children}
          </code>
        );
      }
      
      return (
        <div className="md-code-block">
          {language && <span className="md-code-lang">{language}</span>}
          <pre>
            <code className={className} {...props}>
              {children}
            </code>
          </pre>
        </div>
      );
    },
    // Tables
    table: ({ children }: any) => (
      <div className="md-table-wrapper">
        <table className="md-table">{children}</table>
      </div>
    ),
    // Links
    a: ({ href, children }: any) => (
      <a href={href} target="_blank" rel="noopener noreferrer" className="md-link">
        {children}
      </a>
    ),
    // Blockquotes
    blockquote: ({ children }: any) => (
      <blockquote className="md-blockquote">{children}</blockquote>
    ),
    // Lists
    ul: ({ children }: any) => <ul className="md-list md-list--unordered">{children}</ul>,
    ol: ({ children }: any) => <ol className="md-list md-list--ordered">{children}</ol>,
    li: ({ children }: any) => <li className="md-list-item">{children}</li>,
    // Headings
    h1: ({ children }: any) => <h1 className="md-heading md-h1">{children}</h1>,
    h2: ({ children }: any) => <h2 className="md-heading md-h2">{children}</h2>,
    h3: ({ children }: any) => <h3 className="md-heading md-h3">{children}</h3>,
    h4: ({ children }: any) => <h4 className="md-heading md-h4">{children}</h4>,
    // Paragraphs
    p: ({ children }: any) => <p className="md-paragraph">{children}</p>,
    // Horizontal rule
    hr: () => <hr className="md-hr" />,
    // Strong/Bold
    strong: ({ children }: any) => <strong className="md-strong">{children}</strong>,
    // Emphasis/Italic
    em: ({ children }: any) => <em className="md-em">{children}</em>,
  }), []);

  return (
    <div className={`message message--${role}`}>
      {role === 'assistant' && (
        <div className="message__avatar">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 16v-4" />
            <path d="M12 8h.01" />
          </svg>
        </div>
      )}
      <div className={`message__bubble message__bubble--${role}`}>
        {isLoading || (isStreaming && !content && (!toolCalls || toolCalls.length === 0)) ? (
          <div className="message__loading">
            <span className="message__loading-dot"></span>
            <span className="message__loading-dot"></span>
            <span className="message__loading-dot"></span>
          </div>
        ) : (
          <>
            {/* Tool calls rendered before content */}
            {toolCalls && toolCalls.length > 0 && (
              <div className="message__tool-calls">
                {toolCalls.map((toolCall) => (
                  <CardRenderer
                    key={toolCall.id}
                    toolName={toolCall.name}
                    toolOutput={toolCall.result || ''}
                    status={toolCall.status}
                  />
                ))}
              </div>
            )}
            
            {/* Markdown content */}
            {content && (
              <div className="message__markdown">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={markdownComponents}
                >
                  {content}
                </ReactMarkdown>
                {isStreaming && <span className="message__cursor">▊</span>}
              </div>
            )}
            
            {/* Timestamp */}
            {timestamp && !isStreaming && (
              <span className="message__timestamp">
                {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default Message;
