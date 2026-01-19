# Dynamic UI Cards - Visual Reference Guide

## Card Layouts

### 🔵 CallCard (Outgoing Call Banner)
```
┌─────────────────────────────────────────┐
│ [Sage→Slate Blue Gradient Background]  │
│                                         │
│  ┌──────┐  CALLING (pulsing)          │
│  │ 📞  │  John Smith                   │
│  │ ⚪  │  Receptionist                 │
│  └──────┘  (555) 123-4567              │
│   ◯  ◯    0:45                         │
│                                         │
└─────────────────────────────────────────┘
```
- Phone icon with pulse animation
- Expanding rings during active calls
- Status, Name, Role, Phone, Duration

---

### 🔵 PatientProfileCard
```
┌─────────────────────────────────────────┐
│ [Sage→Slate Blue Gradient Background]  │
│                                         │
│  Homer Simpson          65 years old    │
│  ─────────────────────────────────────  │
│  DOB        Jan 15, 1959               │
│  Email      homer@example.com          │
│  Phone      (555) 123-4567             │
│                                         │
│  RECENT NOTES                           │
│  ├─ Patient reports headache...        │
│  │  Dr. Smith • Jan 18, 2026          │
│  └─ Follow-up scheduled...             │
│     Dr. Jones • Jan 17, 2026           │
│                                         │
│  UPCOMING APPOINTMENTS                  │
│  Jan 20, 2026    [confirmed]           │
│  Jan 25, 2026    [pending]             │
│                                         │
└─────────────────────────────────────────┘
```
- Full patient details
- Clinical notes with left border
- Appointments with status badges

---

### 🔵 PatientListCard
```
┌─────────────────────────────────────────┐
│ [Sage→Slate Blue Gradient Background]  │
│                                         │
│  Patients                    12 TOTAL   │
│  ─────────────────────────────────────  │
│                                         │
│  ├─ Homer Simpson      [65 yrs]        │
│  │  homer@example.com                  │
│  │                                      │
│  ├─ Marge Simpson      [63 yrs]        │
│  │  marge@example.com                  │
│  │                                      │
│  └─ Bart Simpson       [12 yrs]        │
│     bart@example.com                   │
│                                         │
└─────────────────────────────────────────┘
```
- Compact list layout
- Age badges
- Hover effects

---

### 🔴 LabResultCard (ABNORMAL)
```
┌─────────────────────────────────────────┐
│ [Sage→Slate Blue Gradient] [RED BORDER]│
│                                         │
│  Glucose              [ABNORMAL] ⚠️    │
│  ─────────────────────────────────────  │
│                                         │
│  🔴 185  mg/dL                         │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ NORMAL RANGE                    │   │
│  │ 70 - 100 mg/dL                  │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ├─ NOTES:                             │
│  │  Patient was not fasting           │
│  └─────────────────────────────────    │
│                                         │
│  Jan 19, 2026 10:30 AM                 │
│                                         │
└─────────────────────────────────────────┘
```
- **RED border for abnormal values**
- **RED pulsing ABNORMAL badge**
- **RED value text (185)**
- Large value display (36px)
- Normal range reference

---

### 🟢 LabResultCard (NORMAL)
```
┌─────────────────────────────────────────┐
│ [Sage→Slate Blue Gradient]             │
│                                         │
│  Hemoglobin                             │
│  ─────────────────────────────────────  │
│                                         │
│  ⚪ 14.5  g/dL                         │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ NORMAL RANGE                    │   │
│  │ 13.5 - 17.5 g/dL                │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Jan 19, 2026 10:30 AM                 │
│                                         │
└─────────────────────────────────────────┘
```
- White value text (normal)
- Sage green left border
- Clean, minimal design

---

### 🔵 LabResultsListCard
```
┌─────────────────────────────────────────┐
│ [Sage→Slate Blue Gradient Background]  │
│                                         │
│  Lab Results    8 TOTAL  [2 ABNORMAL]  │
│  ─────────────────────────────────────  │
│                                         │
│  🔴 Glucose                    [!]      │
│      185  mg/dL                         │
│      Normal: 70-100 mg/dL  Jan 19      │
│                                         │
│  ⚪ Hemoglobin                          │
│      14.5  g/dL                         │
│      Normal: 13.5-17.5 g/dL  Jan 19    │
│                                         │
│  🔴 Cholesterol                [!]      │
│      245  mg/dL                         │
│      Normal: 125-200 mg/dL  Jan 19     │
│                                         │
└─────────────────────────────────────────┘
```
- Abnormal count badge in header
- Red highlighting for abnormal results
- Exclamation badge (!)
- Compact list format

---

### 🔵 LabOrderCard
```
┌─────────────────────────────────────────┐
│ [Sage→Slate Blue Gradient Background]  │
│                                         │
│  Complete Blood Count    [URGENT] ⚠️   │
│  ─────────────────────────────────────  │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Status      [in progress] 🟡    │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Ordered      Jan 19, 2026 9:00 AM     │
│                                         │
└─────────────────────────────────────────┘
```
- Priority badges (normal/urgent/stat)
- Status badges with colors
- Timeline display

---

### 🔵 LabOrderListCard
```
┌─────────────────────────────────────────┐
│ [Sage→Slate Blue Gradient Background]  │
│                                         │
│  Lab Orders                  5 ORDERS   │
│  ─────────────────────────────────────  │
│                                         │
│  ├─ Complete Blood Count  [URGENT]     │
│  │  [in progress] 🟡      Jan 19       │
│  │                                      │
│  ├─ Lipid Panel          [NORMAL]      │
│  │  [ordered] 🔵          Jan 19        │
│  │                                      │
│  └─ HbA1c                [STAT] 🔴     │
│     [ordered] 🔵          Jan 19        │
│                                         │
└─────────────────────────────────────────┘
```
- Priority badges (color-coded)
- Status badges
- Compact list layout

---

## Color Legend

### Status Colors
- 🟢 **Confirmed/Completed**: Sage Green (#5D8570)
- 🟡 **Pending/In Progress**: Yellow (#FFC107)
- 🔵 **Ordered**: Slate Blue (#647D94)
- 🔴 **Cancelled/Abnormal**: Red (#F44336)

### Priority Colors
- 🟢 **Normal**: Sage Green
- 🟡 **Urgent**: Yellow
- 🔴 **STAT**: Red

### Text Colors
- **Primary**: White (#FFFFFF)
- **Secondary**: Sage 100 (#E3EDE7)
- **Muted**: Slate Blue 500 (#647D94)
- **Abnormal**: Red (#F44336)

---

## Animations

### CallCard
- ✨ Phone icon pulse (scale 1 → 1.05)
- ✨ Expanding rings (2 rings, staggered)
- ✨ Status text pulse (opacity 1 → 0.6)

### LabResultCard (Abnormal)
- ✨ ABNORMAL badge pulse (opacity 1 → 0.7)

### All Cards
- ✨ Fade in on mount (0.3s)
- ✨ Hover translate (2px right)
- ✨ Background transition on hover

---

## Design Principles

1. **Minimalist**: Clean layouts, no clutter
2. **Sharp**: Clear typography, defined borders
3. **Informative**: All critical data visible
4. **Scannable**: Visual hierarchy with size/color
5. **Responsive**: Works on mobile/tablet/desktop
6. **Accessible**: High contrast, semantic HTML
7. **Branded**: Consistent Sage Green & Slate Blue

---

## Usage Examples

### Normal Lab Result
```typescript
{
  testName: "Hemoglobin",
  value: 14.5,
  unit: "g/dL",
  normalRangeMin: 13.5,
  normalRangeMax: 17.5,
  isAbnormal: false  // ✅ Normal - white text, sage border
}
```

### Abnormal Lab Result
```typescript
{
  testName: "Glucose",
  value: 185,
  unit: "mg/dL",
  normalRangeMin: 70,
  normalRangeMax: 100,
  isAbnormal: true  // 🔴 Abnormal - RED text, RED border, pulsing badge
}
```

### Urgent Lab Order
```typescript
{
  testType: "Complete Blood Count",
  priority: "urgent",  // 🟡 Yellow badge
  status: "in_progress"  // 🟡 Yellow status
}
```

---

All cards follow the same design system for consistency and professional appearance!
