/**
 * StaffProfileCard Component
 * Displays staff profile information with brand colors
 */

import React from 'react';
import { StaffProfileCardProps } from '../types';
import './StaffProfileCard.css';

/**
 * Get display text for staff status
 */
function getStatusText(status: string): string {
  switch (status) {
    case 'available':
      return 'Available';
    case 'busy':
      return 'Busy';
    case 'offline':
      return 'Offline';
    default:
      return status;
  }
}

/**
 * StaffProfileCard - Staff profile and status display
 */
export const StaffProfileCard: React.FC<StaffProfileCardProps> = ({ data }) => {
  const { name, role, phoneNumber, status, department, recentTasks } = data;

  const statusText = getStatusText(status);
  const isAvailable = status === 'available';

  return (
    <div className="staff-card">
      <div className="staff-card__content">
        <h3 className="staff-card__name">{name}</h3>
        <p className="staff-card__role">{role}</p>
        <p className="staff-card__phone">{phoneNumber}</p>
        
        {department && (
          <p className="staff-card__department">{department}</p>
        )}
        
        <p className={`staff-card__status ${isAvailable ? 'staff-card__status--available' : ''}`}>
          {statusText}
        </p>
        
        {recentTasks && recentTasks.length > 0 && (
          <div className="staff-card__tasks">
            <span className="staff-card__tasks-label">Recent Tasks</span>
            {recentTasks.slice(0, 3).map((task) => (
              <div key={task.id} className="staff-card__task">
                {task.title}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default StaffProfileCard;
