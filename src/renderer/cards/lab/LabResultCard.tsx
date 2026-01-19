/**
 * LabResultCard Component
 * Displays lab test result with abnormal value flagging
 */

import React from 'react';
import { BaseCardProps } from '../types';
import './LabResultCard.css';

interface LabResultData {
  id: string;
  labOrderId: string;
  testName: string;
  value: number;
  unit: string;
  normalRangeMin: number;
  normalRangeMax: number;
  isAbnormal: boolean;
  notes?: string;
  createdAt: string;
}

export interface LabResultCardProps extends BaseCardProps<LabResultData> {}

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

export const LabResultCard: React.FC<LabResultCardProps> = ({ data }) => {
  const { testName, value, unit, normalRangeMin, normalRangeMax, isAbnormal, notes, createdAt } = data;

  return (
    <div className={`lab-result-card ${isAbnormal ? 'lab-result-card--abnormal' : ''}`}>
      <div className="lab-result-card__header">
        <h3 className="lab-result-card__test-name">{testName}</h3>
        {isAbnormal && (
          <span className="lab-result-card__abnormal-badge">ABNORMAL</span>
        )}
      </div>

      <div className="lab-result-card__value-section">
        <div className="lab-result-card__value-main">
          <span className={`lab-result-card__value ${isAbnormal ? 'lab-result-card__value--abnormal' : ''}`}>
            {value}
          </span>
          <span className="lab-result-card__unit">{unit}</span>
        </div>
        <div className="lab-result-card__range">
          <span className="lab-result-card__range-label">Normal Range</span>
          <span className="lab-result-card__range-value">
            {normalRangeMin} - {normalRangeMax} {unit}
          </span>
        </div>
      </div>

      {notes && (
        <div className="lab-result-card__notes">
          <span className="lab-result-card__notes-label">Notes:</span>
          <p className="lab-result-card__notes-text">{notes}</p>
        </div>
      )}

      <div className="lab-result-card__footer">
        <span className="lab-result-card__timestamp">{formatDate(createdAt)}</span>
      </div>
    </div>
  );
};

export default LabResultCard;
