import React, { useState } from 'react';
import './ToolCallCard.css';

export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
  result?: string | any; // Can be string or parsed object
  status: 'pending' | 'success' | 'error';
}

interface ToolCallCardProps {
  toolCall: ToolCall;
}

/**
 * ToolCallCard Component - Clinical Zen Tool Call Display
 * Collapsible card displaying AI tool invocations and results
 */
export const ToolCallCard: React.FC<ToolCallCardProps> = ({ toolCall }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getStatusIcon = () => {
    switch (toolCall.status) {
      case 'pending':
        return (
          <svg className="tool-card__status-icon tool-card__status-icon--pending" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
        );
      case 'success':
        return (
          <svg className="tool-card__status-icon tool-card__status-icon--success" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
        );
      case 'error':
        return (
          <svg className="tool-card__status-icon tool-card__status-icon--error" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
        );
    }
  };

  const getResultPreview = () => {
    if (!toolCall.result) return 'Running...';
    
    // Handle result as object or string
    const resultStr = typeof toolCall.result === 'string' 
      ? toolCall.result 
      : JSON.stringify(toolCall.result);
    
    const maxLength = 50;
    if (resultStr.length <= maxLength) return resultStr;
    return resultStr.slice(0, maxLength) + '...';
  };

  return (
    <div className="tool-card">
      <button 
        className="tool-card__header"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <div className="tool-card__header-left">
          <span className="tool-card__name">{toolCall.name}</span>
          <span className="tool-card__status">
            {getStatusIcon()}
            <span className="tool-card__status-text">
              {toolCall.status === 'pending' ? 'Running' : toolCall.status === 'success' ? 'Complete' : 'Failed'}
            </span>
          </span>
        </div>
        <svg 
          className={`tool-card__chevron ${isExpanded ? 'tool-card__chevron--expanded' : ''}`}
          width="16" 
          height="16" 
          viewBox="0 0 24 24" 
          fill="none" 
          stroke="currentColor" 
          strokeWidth="2"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      
      {!isExpanded && toolCall.result && (
        <div className="tool-card__preview">
          {getResultPreview()}
        </div>
      )}
      
      <div className={`tool-card__body ${isExpanded ? '' : 'tool-card__body--collapsed'}`}>
        <div className="tool-card__section">
          <span className="tool-card__section-label">Arguments</span>
          <pre className="tool-card__code">{toolCall.arguments}</pre>
        </div>
        {toolCall.result && (
          <div className="tool-card__section">
            <span className="tool-card__section-label">Result</span>
            <pre className="tool-card__code">
              {typeof toolCall.result === 'string' 
                ? toolCall.result 
                : JSON.stringify(toolCall.result, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default ToolCallCard;
