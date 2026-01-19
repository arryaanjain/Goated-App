/**
 * StaffListCard Component
 * Displays list of staff members with brand colors
 */

import React from 'react';
import { StaffListCardProps } from '../types';
import './StaffListCard.css';

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
 * Format role for display
 */
function formatRole(role: string): string {
  return role.charAt(0).toUpperCase() + role.slice(1);
}

/**
 * StaffListCard - Staff roster display
 */
export const StaffListCard: React.FC<StaffListCardProps> = ({ data }) => {
  const { staff, count } = data;

  if (!staff || staff.length === 0) {
    return (
      <div className="staff-list-card">
        <div className="staff-list-card__content">
          <p className="staff-list-card__empty">No staff members found</p>
        </div>
      </div>
    );
  }

  return (
    <div className="staff-list-card">
      <div className="staff-list-card__content">
        <h3 className="staff-list-card__title">Staff Roster ({count || staff.length})</h3>
        <div className="staff-list-card__list">
          {staff.map((member) => {
            const isAvailable = member.status === 'available';
            const taskCount = member.tasks?.length || 0;

            return (
              <div key={member.id} className="staff-list-card__member">
                <div className="staff-list-card__member-header">
                  <div className="staff-list-card__member-info">
                    <span className="staff-list-card__member-name">{member.name}</span>
                    <span className="staff-list-card__member-role">{formatRole(member.role)}</span>
                  </div>
                  <span className={`staff-list-card__status ${isAvailable ? 'staff-list-card__status--available' : ''}`}>
                    {getStatusText(member.status)}
                  </span>
                </div>
                <div className="staff-list-card__member-details">
                  <span className="staff-list-card__member-phone">{member.phone}</span>
                  {taskCount > 0 && (
                    <span className="staff-list-card__member-tasks">
                      {taskCount} {taskCount === 1 ? 'task' : 'tasks'}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default StaffListCard;
