# Quick Card Reference

## 🎯 Which Card for Which Tool?

### 📞 Calls
```
call_staff              → CallCard
call_staff_by_name      → CallCard
```

### 🏥 Patients
```
get_patient_info        → PatientInfoCard (basic)
lookup_patient          → PatientInfoCard (basic)
search_patient          → PatientInfoCard (basic)

get_patient_profile     → PatientProfileCard (full with notes)
create_patient          → PatientProfileCard (full with notes)

list_patients           → PatientListCard (multiple)
```

### 👥 Staff
```
get_staff_profile       → StaffProfileCard (single)
create_staff            → StaffProfileCard (single)
update_staff_status     → StaffProfileCard (single)

list_staff              → StaffListCard (multiple)
find_available_staff    → StaffListCard (multiple)
find_staff_by_name      → StaffListCard (multiple)
```

### 📋 Tasks
```
list_tasks              → TaskListCard
list_active_calls       → TaskListCard
assign_task             → TaskListCard
complete_task           → TaskListCard
get_task_status         → TaskListCard
get_call_result         → TaskListCard
```

### 🧪 Lab Results
```
create_lab_result       → LabResultCard (single)
get_result_details      → LabResultCard (single)

get_lab_results         → LabResultsListCard (multiple)
```

### 📝 Lab Orders
```
order_lab_test          → LabOrderCard (single)
get_order_status        → LabOrderCard (single)
update_order_status     → LabOrderCard (single)

list_lab_orders         → LabOrderListCard (multiple)
list_pending_orders     → LabOrderListCard (multiple)
```

---

## 🎨 Color Quick Reference

### Status Colors
- **Confirmed/Completed**: `#5D8570` (Sage Green)
- **Pending/In Progress**: `#FFC107` (Yellow)
- **Ordered**: `#647D94` (Slate Blue)
- **Cancelled/Abnormal**: `#F44336` (Red)

### Priority Colors
- **Normal**: `#5D8570` (Sage Green)
- **Urgent**: `#FFC107` (Yellow)
- **STAT**: `#F44336` (Red)

### Text Colors
- **Primary**: `#FFFFFF` (White)
- **Secondary**: `#E3EDE7` (Sage 100)
- **Muted**: `#647D94` (Slate Blue 500)
- **Abnormal**: `#F44336` (Red)

---

## 🔥 Special Features

### Abnormal Lab Values
- Automatically detected via `isAbnormal: true`
- RED border (4px left)
- RED pulsing badge
- RED value text
- RED glow shadow

### Call Animations
- Phone icon pulse (active calls)
- Expanding rings (2 rings)
- Status text pulse

### Priority Badges
- Normal: Green background
- Urgent: Yellow background
- STAT: Red background

---

## 📱 Responsive Breakpoints

```css
/* Mobile */
@media (max-width: 640px) {
  padding: 20px;
  font-size: smaller;
}

/* Tablet */
@media (min-width: 641px) and (max-width: 1024px) {
  padding: 22px;
}

/* Desktop */
@media (min-width: 1025px) {
  padding: 24px;
}
```

---

## 🚀 Quick Start

1. **Import the card**:
```typescript
import { LabResultCard } from '@/cards/lab/LabResultCard';
```

2. **Use in CardRenderer**:
```typescript
// Automatically handled by CardRenderer
<CardRenderer 
  toolName="get_lab_results"
  toolOutput={jsonString}
  status="success"
/>
```

3. **Data format**:
```typescript
{
  success: true,
  data: {
    // Your card-specific data
  },
  metadata: {
    componentType: 'LabResultCard'
  }
}
```

---

## ✅ All Cards Implemented

- [x] CallCard
- [x] PatientInfoCard
- [x] PatientProfileCard ✅ NEW
- [x] PatientListCard ✅ NEW
- [x] StaffProfileCard
- [x] StaffListCard
- [x] TaskListCard
- [x] LabResultCard ✅ NEW
- [x] LabResultsListCard ✅ NEW
- [x] LabOrderCard ✅ NEW
- [x] LabOrderListCard ✅ NEW
- [x] ErrorCard

**Total: 12 card types, 30+ tools registered**
