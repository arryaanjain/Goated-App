/**
 * PatientInfoCard Component
 * Displays patient information with brand colors
 */

import React from 'react';
import { PatientInfoCardProps } from '../types';
import './PatientInfoCard.css';

/**
 * PatientInfoCard - Patient demographic and medical information display
 */
export const PatientInfoCard: React.FC<PatientInfoCardProps> = ({ data }) => {
  const { name, mrn, age, dob, gender, room, allergies, conditions } = data;

  return (
    <div className="patient-card">
      <div className="patient-card__content">
        <h3 className="patient-card__name">{name}</h3>
        <p className="patient-card__mrn">MRN: {mrn}</p>
        
        <div className="patient-card__demographics">
          <span>{age}</span>
          <span>{gender}</span>
          <span>{dob}</span>
        </div>
        
        {room && (
          <p className="patient-card__room">Room {room}</p>
        )}
        
        {allergies && allergies.length > 0 && (
          <div className="patient-card__allergies">
            <span className="patient-card__warning-icon">⚠</span>
            <span>{allergies.join(', ')}</span>
          </div>
        )}
        
        {conditions && conditions.length > 0 && (
          <div className="patient-card__conditions">
            {conditions.map((condition, index) => (
              <span key={index} className="patient-card__condition">
                {condition}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default PatientInfoCard;
