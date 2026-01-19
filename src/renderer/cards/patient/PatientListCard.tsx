/**
 * PatientListCard Component
 * Displays a list of patients
 */

import React from 'react';
import { BaseCardProps } from '../types';
import './PatientListCard.css';

interface Patient {
  id: string;
  name: string;
  email: string;
  dateOfBirth: string;
}

interface PatientListData {
  patients: Patient[];
  total: number;
  page: number;
  limit: number;
}

export interface PatientListCardProps extends BaseCardProps<PatientListData> {}

function calculateAge(dob: string): number {
  const birthDate = new Date(dob);
  const today = new Date();
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
    age--;
  }
  return age;
}

export const PatientListCard: React.FC<PatientListCardProps> = ({ data }) => {
  const { patients, total } = data;

  return (
    <div className="patient-list-card">
      <div className="patient-list-card__header">
        <h3 className="patient-list-card__title">Patients</h3>
        <span className="patient-list-card__count">{total} total</span>
      </div>

      <div className="patient-list-card__list">
        {patients.map((patient) => (
          <div key={patient.id} className="patient-list-card__item">
            <div className="patient-list-card__item-main">
              <span className="patient-list-card__item-name">{patient.name}</span>
              <span className="patient-list-card__item-age">{calculateAge(patient.dateOfBirth)} yrs</span>
            </div>
            <span className="patient-list-card__item-email">{patient.email}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PatientListCard;
