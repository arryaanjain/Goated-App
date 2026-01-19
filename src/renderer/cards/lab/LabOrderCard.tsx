/**
 * LabOrderCard Component
 * Displays lab order details
 */

import React from 'react';
import { BaseCardProps } from '../types';
import './LabOrderCard.css';

interface LabOrderData {
  id: string;
  patientId: string;
  testType: string;
  priority: 'normal' | 'urgent' | 'stat';
  status: 'ordered' | 'in_progress' | 'completed';
  orderedAt: string;
  completedAt?: string;
}

export interface LabOrderCardProps extends BaseCardProps<LabOrderData> {}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { 
    month: 'short', 
    day: 'numeric', 
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

export const LabOrderCard: React.FC<LabOrderCardProps> = ({ data }) => {
  const { testType, priority, status, orderedAt, completedAt } = data;

  return (
    <div className="lab-order-card">
      <div className="lab-order-card__header">
        <h3 className="lab-order-card__test-type">{testType}</h3>
        <span className={`lab-order-card__priority lab-order-card__priority--${priority}`}>
          {priority}
        </span>
      </div>

      <div className="lab-order-card__status-section">
        <span className="lab-order-card__status-label">Status</span>
        <span className={`lab-order-card__status lab-order-card__status--${status}`}>
          {status.replace('_', ' ')}
        </span>
      </div>

      <div className="lab-order-card__timeline">
        <div className="lab-order-card__timeline-item">
          <span className="lab-order-card__timeline-label">Ordered</span>
          <span className="lab-order-card__timeline-value">{formatDate(orderedAt)}</span>
        </div>
        {completedAt && (
          <div className="lab-order-card__timeline-item">
            <span className="lab-order-card__timeline-label">Completed</span>
            <span className="lab-order-card__timeline-value">{formatDate(completedAt)}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default LabOrderCard;
