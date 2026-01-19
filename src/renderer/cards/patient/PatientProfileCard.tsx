/**
 * PatientProfileCard Component
 * Full patient profile with clinical notes and appointments
 */

import React from 'react';
import { BaseCardProps } from '../types';
import './PatientProfileCard.css';

interface ClinicalNote {
  id: string;
  text: string;
  createdBy: string;
  createdAt: string;
}

interface Appointment {
  id: string;
  doctorId: string;
  scheduledTime: string;
  status: string;
  notes?: string;
}

interface PatientProfileData {
  id: string;
  name: string;
  email: string;
  phone?: string;
  dateOfBirth: string;
  createdAt: string;
  clinicalNotes?: ClinicalNote[];
  appointments?: Appointment[];
}

export interface PatientProfileCardProps extends BaseCardProps<PatientProfileData> {}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

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

export const PatientProfileCard: React.FC<PatientProfileCardProps> = ({ data }) => {
  const { name, email, phone, dateOfBirth, clinicalNotes, appointments } = data;
  const age = calculateAge(dateOfBirth);

  return (
    <div className="patient-profile-card">
      <div className="patient-profile-card__header">
        <h3 className="patient-profile-card__name">{name}</h3>
        <p className="patient-profile-card__age">{age} years old</p>
      </div>

      <div className="patient-profile-card__info">
        <div className="patient-profile-card__info-item">
          <span className="patient-profile-card__label">DOB</span>
          <span className="patient-profile-card__value">{formatDate(dateOfBirth)}</span>
        </div>
        <div className="patient-profile-card__info-item">
          <span className="patient-profile-card__label">Email</span>
          <span className="patient-profile-card__value">{email}</span>
        </div>
        {phone && (
          <div className="patient-profile-card__info-item">
            <span className="patient-profile-card__label">Phone</span>
            <span className="patient-profile-card__value">{phone}</span>
          </div>
        )}
      </div>

      {clinicalNotes && clinicalNotes.length > 0 && (
        <div className="patient-profile-card__section">
          <h4 className="patient-profile-card__section-title">Recent Notes</h4>
          <div className="patient-profile-card__notes">
            {clinicalNotes.slice(0, 3).map((note) => (
              <div key={note.id} className="patient-profile-card__note">
                <p className="patient-profile-card__note-text">{note.text}</p>
                <p className="patient-profile-card__note-meta">
                  {note.createdBy} • {formatDate(note.createdAt)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {appointments && appointments.length > 0 && (
        <div className="patient-profile-card__section">
          <h4 className="patient-profile-card__section-title">Upcoming Appointments</h4>
          <div className="patient-profile-card__appointments">
            {appointments.slice(0, 3).map((apt) => (
              <div key={apt.id} className="patient-profile-card__appointment">
                <span className="patient-profile-card__appointment-date">
                  {formatDate(apt.scheduledTime)}
                </span>
                <span className={`patient-profile-card__appointment-status patient-profile-card__appointment-status--${apt.status}`}>
                  {apt.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PatientProfileCard;
