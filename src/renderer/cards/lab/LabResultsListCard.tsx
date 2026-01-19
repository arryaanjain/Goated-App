/**
 * LabResultsListCard Component
 * Displays list of lab results with abnormal value highlighting
 */

import React from 'react';
import { BaseCardProps } from '../types';
import './LabResultsListCard.css';

interface LabResult {
  id: string;
  testName: string;
  value: number;
  unit: string;
  normalRangeMin: number;
  normalRangeMax: number;
  isAbnormal: boolean;
  createdAt: string;
}

interface LabResultsListData {
  results: LabResult[];
  patientId: string;
}

export interface LabResultsListCardProps extends BaseCardProps<LabResultsListData> {}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export const LabResultsListCard: React.FC<LabResultsListCardProps> = ({ data }) => {
  const { results } = data;
  const abnormalCount = results.filter(r => r.isAbnormal).length;

  return (
    <div className="lab-results-list-card">
      <div className="lab-results-list-card__header">
        <h3 className="lab-results-list-card__title">Lab Results</h3>
        <div className="lab-results-list-card__counts">
          <span className="lab-results-list-card__count">{results.length} total</span>
          {abnormalCount > 0 && (
            <span className="lab-results-list-card__abnormal-count">{abnormalCount} abnormal</span>
          )}
        </div>
      </div>

      <div className="lab-results-list-card__list">
        {results.map((result) => (
          <div 
            key={result.id} 
            className={`lab-results-list-card__item ${result.isAbnormal ? 'lab-results-list-card__item--abnormal' : ''}`}
          >
            <div className="lab-results-list-card__item-header">
              <span className="lab-results-list-card__item-name">{result.testName}</span>
              {result.isAbnormal && (
                <span className="lab-results-list-card__item-badge">!</span>
              )}
            </div>
            <div className="lab-results-list-card__item-value">
              <span className={`lab-results-list-card__item-number ${result.isAbnormal ? 'lab-results-list-card__item-number--abnormal' : ''}`}>
                {result.value}
              </span>
              <span className="lab-results-list-card__item-unit">{result.unit}</span>
            </div>
            <div className="lab-results-list-card__item-footer">
              <span className="lab-results-list-card__item-range">
                Normal: {result.normalRangeMin}-{result.normalRangeMax} {result.unit}
              </span>
              <span className="lab-results-list-card__item-date">{formatDate(result.createdAt)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default LabResultsListCard;
