/**
 * Card Registry System
 * Maps MCP tool names to their corresponding UI card components
 */

import { CardRegistry, CardComponent } from './types';
import { CallCard } from './call/CallCard';
import { PatientInfoCard } from './patient/PatientInfoCard';
import { PatientProfileCard } from './patient/PatientProfileCard';
import { PatientListCard } from './patient/PatientListCard';
import { StaffProfileCard } from './staff/StaffProfileCard';
import { StaffListCard } from './staffList/StaffListCard';
import { TaskListCard } from './task/TaskListCard';
import { LabResultCard } from './lab/LabResultCard';
import { LabResultsListCard } from './lab/LabResultsListCard';
import { LabOrderCard } from './lab/LabOrderCard';
import { LabOrderListCard } from './lab/LabOrderListCard';
import { ErrorCard } from './error/ErrorCard';

// Central card registry
export const cardRegistry: CardRegistry = {
  // Call-related tools
  'call_staff': CallCard,
  'call_staff_by_name': CallCard,
  
  // Patient-related tools (basic info)
  'get_patient_info': PatientInfoCard,
  'lookup_patient': PatientInfoCard,
  'search_patient': PatientInfoCard,
  
  // Patient-related tools (full profile)
  'get_patient_profile': PatientProfileCard,
  'create_patient': PatientProfileCard,
  
  // Patient-related tools (list)
  'list_patients': PatientListCard,
  
  // Staff-related tools (single staff member)
  'get_staff_profile': StaffProfileCard,
  'create_staff': StaffProfileCard,
  'update_staff_status': StaffProfileCard,
  
  // Staff-related tools (multiple staff members)
  'list_staff': StaffListCard,
  'find_available_staff': StaffListCard,
  'find_staff_by_name': StaffListCard,
  
  // Task-related tools
  'list_tasks': TaskListCard,
  'list_active_calls': TaskListCard,
  'assign_task': TaskListCard,
  'complete_task': TaskListCard,
  'get_task_status': TaskListCard,
  'get_call_result': TaskListCard,
  
  // Lab test tools (single result)
  'create_lab_result': LabResultCard,
  'get_result_details': LabResultCard,
  
  // Lab test tools (results list)
  'get_lab_results': LabResultsListCard,
  
  // Lab order tools (single order)
  'order_lab_test': LabOrderCard,
  'get_order_status': LabOrderCard,
  'update_order_status': LabOrderCard,
  
  // Lab order tools (order list)
  'list_lab_orders': LabOrderListCard,
  'list_pending_orders': LabOrderListCard,
  
  // Error handling
  'error': ErrorCard,
};

/**
 * Look up a card component by tool name
 * @param toolName - The name of the MCP tool
 * @returns The card component or null if not found
 */
export function getCardForTool(toolName: string): CardComponent | null {
  return cardRegistry[toolName] || null;
}

/**
 * Register a new card component for a tool
 * @param toolName - The name of the MCP tool
 * @param component - The React component to render
 */
export function registerCard(toolName: string, component: CardComponent): void {
  cardRegistry[toolName] = component;
}

/**
 * Check if a tool has a registered card
 * @param toolName - The name of the MCP tool
 * @returns True if a card is registered for this tool
 */
export function hasCard(toolName: string): boolean {
  return toolName in cardRegistry;
}

// Export card components
export { CallCard } from './call/CallCard';
export { PatientInfoCard } from './patient/PatientInfoCard';
export { PatientProfileCard } from './patient/PatientProfileCard';
export { PatientListCard } from './patient/PatientListCard';
export { StaffProfileCard } from './staff/StaffProfileCard';
export { StaffListCard } from './staffList/StaffListCard';
export { TaskListCard } from './task/TaskListCard';
export { LabResultCard } from './lab/LabResultCard';
export { LabResultsListCard } from './lab/LabResultsListCard';
export { LabOrderCard } from './lab/LabOrderCard';
export { LabOrderListCard } from './lab/LabOrderListCard';
export { ErrorCard } from './error/ErrorCard';
export { CardRenderer } from './CardRenderer';

// Export types
export type { CardRegistry, CardComponent } from './types';
