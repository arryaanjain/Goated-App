/**
 * TypeScript interfaces for Dynamic UI Cards system
 */

// Base props that all cards receive
export interface BaseCardProps<T = any> {
  data: T;
  status: 'pending' | 'success' | 'error';
}

// CardRenderer props
export interface CardRendererProps {
  toolName: string;
  toolOutput: string; // JSON string from MCP tool
  status: 'pending' | 'success' | 'error';
}

// MCP Tool Response Format
export interface MCPResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: {
    componentType?: string;
    timestamp?: string;
    [key: string]: any;
  };
}

// CallCard specific types
export interface MCPCallResponse extends MCPResponse<CallCardData> {
  data: CallCardData;
}

export interface CallCardData {
  staffName: string;
  phoneNumber: string;
  callStatus: 'calling' | 'ringing' | 'connected' | 'not_picked_up' | 'cancelled' | 'ended';
  role?: string; // Staff role (e.g., "Receptionist", "Doctor")
  duration?: number;
  transcript?: string;
}

export interface CallCardProps extends BaseCardProps<CallCardData> {}

// PatientInfoCard specific types
export interface MCPPatientInfoResponse extends MCPResponse<PatientInfoCardData> {
  data: PatientInfoCardData;
}

export interface PatientInfoCardData {
  name: string;
  mrn: string;
  age: number;
  dob: string;
  gender: string;
  room?: string;
  allergies?: string[];
  conditions?: string[];
}

export interface PatientInfoCardProps extends BaseCardProps<PatientInfoCardData> {}

// StaffProfileCard specific types (single staff member)
export interface MCPStaffProfileResponse extends MCPResponse<StaffProfileCardData> {
  data: StaffProfileCardData;
}

export interface StaffProfileCardData {
  name: string;
  role: string;
  phoneNumber: string;
  status: 'available' | 'busy' | 'offline';
  department?: string;
  recentTasks?: Array<{
    id: string;
    title: string;
    timestamp: string;
  }>;
}

export interface StaffProfileCardProps extends BaseCardProps<StaffProfileCardData> {}

// StaffListCard specific types (multiple staff members)
export interface MCPStaffListResponse extends MCPResponse<StaffListCardData> {
  data: StaffListCardData;
}

export interface StaffListCardData {
  staff: Array<{
    id: string;
    name: string;
    phone: string;
    role: string;
    status: 'available' | 'busy' | 'offline';
    createdAt?: string;
    updatedAt?: string;
    tasks?: Array<{
      id: string;
      description: string;
      status: string;
      [key: string]: any;
    }>;
  }>;
  count: number;
}

export interface StaffListCardProps extends BaseCardProps<StaffListCardData> {}

// TaskListCard specific types
export interface MCPTaskListResponse extends MCPResponse<TaskListCardData> {
  data: TaskListCardData;
}

export interface TaskListCardData {
  tasks: Array<{
    id: string;
    title: string;
    status: 'pending' | 'in_progress' | 'completed' | 'cancelled';
    priority: 'low' | 'medium' | 'high' | 'urgent';
    assignedTo?: string;
    timestamp: string;
  }>;
}

export interface TaskListCardProps extends BaseCardProps<TaskListCardData> {}

// ErrorCard specific types
export interface MCPErrorResponse extends MCPResponse {
  success: false;
  error: string;
}

export interface ErrorCardData {
  error: string;
  code?: string;
  details?: string;
}

export interface ErrorCardProps extends BaseCardProps<ErrorCardData> {}

// PatientProfileCard specific types (full profile with notes and appointments)
export interface PatientProfileData {
  id: string;
  name: string;
  email: string;
  phone?: string;
  dateOfBirth: string;
  createdAt: string;
  clinicalNotes?: Array<{
    id: string;
    text: string;
    createdBy: string;
    createdAt: string;
  }>;
  appointments?: Array<{
    id: string;
    doctorId: string;
    scheduledTime: string;
    status: string;
    notes?: string;
  }>;
}

export interface PatientProfileCardProps extends BaseCardProps<PatientProfileData> {}

// PatientListCard specific types
export interface PatientListData {
  patients: Array<{
    id: string;
    name: string;
    email: string;
    dateOfBirth: string;
  }>;
  total: number;
  page: number;
  limit: number;
}

export interface PatientListCardProps extends BaseCardProps<PatientListData> {}

// LabResultCard specific types
export interface LabResultData {
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

// LabResultsListCard specific types
export interface LabResultsListData {
  results: Array<{
    id: string;
    testName: string;
    value: number;
    unit: string;
    normalRangeMin: number;
    normalRangeMax: number;
    isAbnormal: boolean;
    createdAt: string;
  }>;
  patientId: string;
}

export interface LabResultsListCardProps extends BaseCardProps<LabResultsListData> {}

// LabOrderCard specific types
export interface LabOrderData {
  id: string;
  patientId: string;
  testType: string;
  priority: 'normal' | 'urgent' | 'stat';
  status: 'ordered' | 'in_progress' | 'completed';
  orderedAt: string;
  completedAt?: string;
}

export interface LabOrderCardProps extends BaseCardProps<LabOrderData> {}

// LabOrderListCard specific types
export interface LabOrderListData {
  orders: Array<{
    id: string;
    testType: string;
    priority: 'normal' | 'urgent' | 'stat';
    status: 'ordered' | 'in_progress' | 'completed';
    orderedAt: string;
  }>;
}

export interface LabOrderListCardProps extends BaseCardProps<LabOrderListData> {}

// Card Registry types
export type CardComponent = React.ComponentType<BaseCardProps<any>>;

export interface CardRegistry {
  [toolName: string]: CardComponent;
}
