/**
 * TaskListCard Component
 * Displays list of tasks with brand colors
 */

import React from 'react';
import { TaskListCardProps } from '../types';
import './TaskListCard.css';

/**
 * Get display text for task status
 */
function getStatusText(status: string): string {
  switch (status) {
    case 'pending':
      return 'Pending';
    case 'in_progress':
      return 'In Progress';
    case 'completed':
      return 'Completed';
    case 'cancelled':
      return 'Cancelled';
    default:
      return status;
  }
}

/**
 * Get CSS class for priority
 */
function getPriorityClass(priority: string): string {
  return `task-list-card__priority--${priority}`;
}

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();
  
  if (isToday) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

/**
 * TaskListCard - Task list display
 */
export const TaskListCard: React.FC<TaskListCardProps> = ({ data }) => {
  const { tasks } = data;

  if (!tasks || tasks.length === 0) {
    return (
      <div className="task-list-card">
        <div className="task-list-card__content">
          <p className="task-list-card__empty">No tasks found</p>
        </div>
      </div>
    );
  }

  return (
    <div className="task-list-card">
      <div className="task-list-card__content">
        <h3 className="task-list-card__title">Tasks ({tasks.length})</h3>
        <div className="task-list-card__tasks">
          {tasks.map((task) => (
            <div key={task.id} className="task-list-card__task">
              <div className="task-list-card__task-header">
                <span className="task-list-card__task-title">{task.title}</span>
                <span className={`task-list-card__priority ${getPriorityClass(task.priority)}`}>
                  {task.priority}
                </span>
              </div>
              <div className="task-list-card__task-meta">
                <span className={`task-list-card__status task-list-card__status--${task.status}`}>
                  {getStatusText(task.status)}
                </span>
                {task.assignedTo && (
                  <span className="task-list-card__assigned">
                    {task.assignedTo}
                  </span>
                )}
                <span className="task-list-card__time">
                  {formatTimestamp(task.timestamp)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TaskListCard;
