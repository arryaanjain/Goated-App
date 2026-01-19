/**
 * ErrorCard Component
 * Displays error information with brand colors
 */

import React from 'react';
import { ErrorCardProps } from '../types';
import './ErrorCard.css';

/**
 * ErrorCard - Error display
 */
export const ErrorCard: React.FC<ErrorCardProps> = ({ data }) => {
  const { error, code, details } = data;

  return (
    <div className="error-card">
      <div className="error-card__content">
        <div className="error-card__icon">⚠</div>
        <h3 className="error-card__title">Error</h3>
        <p className="error-card__message">{error}</p>
        
        {code && (
          <p className="error-card__code">Code: {code}</p>
        )}
        
        {details && (
          <p className="error-card__details">{details}</p>
        )}
      </div>
    </div>
  );
};

export default ErrorCard;
