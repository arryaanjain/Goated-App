/**
 * CallCard Component
 * Outgoing call banner interface with brand colors
 */

import React from 'react';
import { CallCardProps } from '../types';
import './CallCard.css';

/**
 * Get display text for call status
 */
function getStatusText(status: string): string {
  switch (status) {
    case 'calling':
      return 'Calling';
    case 'ringing':
      return 'Ringing';
    case 'connected':
      return 'Connected';
    case 'not_picked_up':
      return 'No Answer';
    case 'cancelled':
      return 'Cancelled';
    case 'ended':
      return 'Call Ended';
    default:
      return status;
  }
}

/**
 * Check if status is active (should pulse)
 */
function isActiveStatus(status: string): boolean {
  return status === 'calling' || status === 'ringing';
}

/**
 * Format duration for display
 */
function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Format role for display
 */
function formatRole(role?: string): string {
  if (!role) return '';
  return role.charAt(0).toUpperCase() + role.slice(1);
}

/**
 * CallCard - Outgoing call banner interface
 */
export const CallCard: React.FC<CallCardProps> = ({ data }) => {
  const { staffName, phoneNumber, callStatus, role, duration } = data;

  const statusText = getStatusText(callStatus);
  const isActive = isActiveStatus(callStatus);
  const isConnected = callStatus === 'connected';
  const isEnded = callStatus === 'ended' || callStatus === 'not_picked_up' || callStatus === 'cancelled';
  const formattedRole = formatRole(role);

  return (
    <div className={`call-card ${isActive ? 'call-card--active' : ''} ${isConnected ? 'call-card--connected' : ''}`}>
      <div className="call-card__content">
        {/* Phone icon with pulse animation */}
        <div className={`call-card__icon ${isActive ? 'call-card__icon--pulse' : ''}`}>
          <svg className="call-card__icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z" />
          </svg>
          {isActive && (
            <>
              <div className="call-card__icon-ring call-card__icon-ring--1"></div>
              <div className="call-card__icon-ring call-card__icon-ring--2"></div>
            </>
          )}
        </div>
        
        {/* Call information */}
        <div className="call-card__info">
          {/* Status */}
          <p className={`call-card__status ${isActive ? 'call-card__status--active' : ''} ${isEnded ? 'call-card__status--ended' : ''}`}>
            {statusText}
          </p>
          
          {/* Name */}
          <h3 className="call-card__name">{staffName}</h3>
          
          {/* Role (if available) */}
          {formattedRole && (
            <p className="call-card__role">{formattedRole}</p>
          )}
          
          {/* Phone number */}
          <p className="call-card__phone">{phoneNumber}</p>
          
          {/* Duration for connected/ended calls */}
          {duration !== undefined && duration > 0 && (
            <p className="call-card__duration">
              {formatDuration(duration)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default CallCard;
