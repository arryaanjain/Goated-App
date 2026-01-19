/**
 * LabOrderListCard Component
 * Displays list of lab orders
 */

import React from 'react';
import { BaseCardProps } from '../types';
import './LabOrderListCard.css';

interface LabOrder {
  id: string;
  testType: string;
  priority: 'normal' | 'urgent' | 'stat';
  status: 'ordered' | 'in_progress' | 'completed';
  orderedAt: string;
}

interface LabOrderListData {
  orders: LabOrder[];
}

export interface LabOrderListCardProps extends BaseCardProps<LabOrderListData> {}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export const LabOrderListCard: React.FC<LabOrderListCardProps> = ({ data }) => {
  const { orders } = data;

  return (
    <div className="lab-order-list-card">
      <div className="lab-order-list-card__header">
        <h3 className="lab-order-list-card__title">Lab Orders</h3>
        <span className="lab-order-list-card__count">{orders.length} orders</span>
      </div>

      <div className="lab-order-list-card__list">
        {orders.map((order) => (
          <div key={order.id} className="lab-order-list-card__item">
            <div className="lab-order-list-card__item-header">
              <span className="lab-order-list-card__item-name">{order.testType}</span>
              <span className={`lab-order-list-card__item-priority lab-order-list-card__item-priority--${order.priority}`}>
                {order.priority}
              </span>
            </div>
            <div className="lab-order-list-card__item-footer">
              <span className={`lab-order-list-card__item-status lab-order-list-card__item-status--${order.status}`}>
                {order.status.replace('_', ' ')}
              </span>
              <span className="lab-order-list-card__item-date">{formatDate(order.orderedAt)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default LabOrderListCard;
