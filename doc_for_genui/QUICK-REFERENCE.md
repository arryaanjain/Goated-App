# Dynamic UI Cards - Quick Reference Guide

## For Developers

### Adding a New Card Type

**Step 1: Create Component**
```typescript
// src/renderer/cards/appointment/AppointmentCard.tsx
import React from 'react';
import { AppointmentCardProps } from '../types';
import './AppointmentCard.css';

export const AppointmentCard: React.FC<AppointmentCardProps> = ({ data }) => {
  return (
    <div className="appointment-card">
      <div className="appointment-card__content">
        <h3 className="appointment-card__name">{data.patientName}</h3>
        <p className="appointment-card__time">{data.time}</p>
      </div>
    </div>
  );
};
```

**Step 2: Create Styles (Use Brand Colors)**
```css
/* src/renderer/cards/appointment/AppointmentCard.css */
.appointment-card {
  /* Brand gradient */
  background: linear-gradient(135deg, var(--color-sage-900) 0%, var(--color-slate-blue-900) 100%);
  
  /* Standard card styling */
  padding: 24px;
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  margin: 12px 0;
  
  /* Overlay */
  position: relative;
  overflow: hidden;
}

.appointment-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(255, 255, 255, 0.03);
  backdrop-filter: blur(8px);
  border-radius: inherit;
  z-index: 0;
}

.appointment-card__content {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.appointment-card__name {
  font-size: 24px;
  font-weight: 700;
  color: var(--color-surface); /* White */
}

.appointment-card__time {
  font-size: 14px;
  color: var(--color-sage-100); /* Light sage */
}
```

**Step 3: Add TypeScript Interface**
```typescript
// src/renderer/cards/types.ts
export interface AppointmentCardData {
  patientName: string;
  time: string;
  doctorName: string;
  room?: string;
}

export interface AppointmentCardProps extends BaseCardProps<AppointmentCardData> {}
```

**Step 4: Register in Registry**
```typescript
// src/renderer/cards/index.ts
import { AppointmentCard } from './appointment/AppointmentCard';

export const cardRegistry: CardRegistry = {
  // ... existing cards
  'schedule_appointment': AppointmentCard,
  'get_appointment': AppointmentCard,
};

// Export the component
export { AppointmentCard } from './appointment/AppointmentCard';
```

**Done!** The card will now automatically render when the MCP tool executes.

---

## Brand Color Reference

### Quick Copy-Paste

```css
/* Background Gradient (All Cards) */
background: linear-gradient(135deg, var(--color-sage-900) 0%, var(--color-slate-blue-900) 100%);

/* Overlay (All Cards) */
background: rgba(255, 255, 255, 0.03);
backdrop-filter: blur(8px);

/* Primary Text (Names, Titles) */
color: var(--color-surface); /* White */
font-size: 24px;
font-weight: 700;

/* Secondary Text (Details) */
color: var(--color-sage-100); /* Light sage */
font-size: 14px;
font-weight: 400;

/* Status/Accent Text */
color: var(--color-sage-100);
font-size: 16px;
font-weight: 500;
text-shadow: 0 0 8px rgba(93, 133, 112, 0.6);

/* Muted Text */
color: var(--color-slate-blue-500); /* Slate blue */
font-size: 14px;

/* Error Text */
color: var(--color-error); /* Red */
text-shadow: 0 0 8px rgba(239, 68, 68, 0.4);

/* Warning Text */
color: var(--color-warning); /* Amber */
```

### Color Variables

```css
/* Sage Green (Primary Brand) */
--color-sage-50: #F2F7F4;
--color-sage-100: #E3EDE7;
--color-sage-500: #5D8570;
--color-sage-700: #3D5C4C;
--color-sage-900: #2A3D33;

/* Slate Blue (Tech/AI Accent) */
--color-slate-blue-50: #F0F4F8;
--color-slate-blue-500: #647D94;
--color-slate-blue-900: #2A3B4C;

/* Text */
--color-text-primary: #1A1A1A;
--color-text-secondary: #6B6B6B;
--color-surface: #FFFFFF;

/* Status */
--color-success: #5D8570;
--color-error: #EF4444;
--color-warning: #F59E0B;
```

---

## Common Patterns

### Status Badge
```css
.card__status {
  padding: 6px 16px;
  border-radius: var(--radius-md);
  font-size: 16px;
  font-weight: 500;
}

/* Active/Available */
.card__status--active {
  color: var(--color-sage-100);
  background: rgba(93, 133, 112, 0.2);
  text-shadow: 0 0 8px rgba(93, 133, 112, 0.4);
}

/* Inactive */
.card__status--inactive {
  color: var(--color-slate-blue-500);
  background: rgba(100, 125, 148, 0.2);
}
```

### Priority Badge
```css
.card__priority {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  padding: 4px 8px;
  border-radius: var(--radius-sm);
}

.card__priority--low {
  background: rgba(100, 125, 148, 0.2);
  color: var(--color-slate-blue-500);
}

.card__priority--medium {
  background: rgba(93, 133, 112, 0.2);
  color: var(--color-sage-100);
}

.card__priority--high {
  background: rgba(245, 158, 11, 0.2);
  color: var(--color-warning);
}

.card__priority--urgent {
  background: rgba(239, 68, 68, 0.2);
  color: var(--color-error);
  text-shadow: 0 0 6px rgba(239, 68, 68, 0.3);
}
```

### List Item
```css
.card__item {
  background: rgba(255, 255, 255, 0.05);
  border-radius: var(--radius-md);
  padding: 12px;
}

.card__item-title {
  font-size: 15px;
  font-weight: 500;
  color: var(--color-surface);
}

.card__item-detail {
  font-size: 13px;
  color: var(--color-sage-100);
}
```

### Warning Section
```css
.card__warning {
  background: rgba(239, 68, 68, 0.1);
  border-radius: var(--radius-md);
  padding: 8px 12px;
  color: var(--color-error);
  text-shadow: 0 0 8px rgba(239, 68, 68, 0.4);
}

.card__warning-icon {
  font-size: 16px;
}
```

---

## MCP Tool Response Format

Your MCP tool should return JSON in this format:

```json
{
  "success": true,
  "data": {
    // Your card-specific data
    "staffName": "Dr. Sarah Chen",
    "phoneNumber": "555-0123",
    "callStatus": "ringing"
  },
  "metadata": {
    "componentType": "call-card",
    "timestamp": "2026-01-19T14:30:00Z"
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Staff member not found",
  "metadata": {
    "code": "STAFF_NOT_FOUND"
  }
}
```

---

## Testing Your Card

### 1. Create Test MCP Server
```python
# test-server.py
@server.call_tool()
async def test_card(name: str) -> list[types.TextContent]:
    return [
        types.TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "data": {
                    "staffName": name,
                    "phoneNumber": "555-0123",
                    "callStatus": "ringing"
                }
            })
        )
    ]
```

### 2. Connect to Server
- Click "Connect Tool" button
- Enter path to your test server
- Click "Connect"

### 3. Test in Chat
```
User: "Test the call card with John Doe"
AI: [Executes test_card tool]
Result: CallCard renders with brand colors
```

### 4. Verify
- ✅ Card uses Sage/Slate Blue gradient
- ✅ Text colors are correct (white, sage, slate)
- ✅ Spacing and padding match design
- ✅ Card persists after page refresh
- ✅ Responsive on mobile

---

## Troubleshooting

### Card Not Rendering
1. Check tool name is registered in `cardRegistry`
2. Verify MCP tool returns valid JSON
3. Check `success: true` in response
4. Look for errors in browser console

### Wrong Colors
1. Ensure using CSS custom properties (`var(--color-sage-900)`)
2. Check `global.css` is imported
3. Verify gradient uses both sage and slate blue

### Layout Issues
1. Check `position: relative` on card
2. Verify `z-index: 1` on content
3. Ensure overlay has `z-index: 0`

### TypeScript Errors
1. Add interface to `types.ts`
2. Export from `index.ts`
3. Import in component file

---

## Checklist for New Cards

- [ ] Component created in `src/renderer/cards/<domain>/`
- [ ] CSS file with brand colors
- [ ] TypeScript interface in `types.ts`
- [ ] Registered in `cardRegistry`
- [ ] Exported from `index.ts`
- [ ] Uses Sage 900 → Slate Blue 900 gradient
- [ ] White primary text, Sage 100 secondary
- [ ] 24px padding, 12px gaps
- [ ] Responsive breakpoints added
- [ ] Reduced motion support
- [ ] Tested with MCP server
- [ ] Verified persistence

---

## Resources

- **Design System**: `.kiro/specs/dynamic-ui-cards/DESIGN-SYSTEM.md`
- **Brand Colors**: `BRAND-COLORS-IMPLEMENTATION.md`
- **Full Docs**: `IMPLEMENTATION-COMPLETE.md`
- **Visual Reference**: `CALLCARD-VISUAL-REFERENCE.md`
- **Comparison**: `BEFORE-AFTER-COMPARISON.md`

---

**Quick Start**: Copy an existing card (e.g., CallCard), rename it, update the data interface, register it, and you're done!
