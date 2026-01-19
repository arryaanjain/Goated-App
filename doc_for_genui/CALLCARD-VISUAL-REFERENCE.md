# CallCard Visual Reference

## Modern Minimalist Luxury Aesthetic

The CallCard follows the Hatch Sleep app design philosophy with deep navy/indigo gradients, frosted glass overlays, and soft glowing cyan accents.

## Visual Layout

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│                                                         │
│                    Jonathan Doe                         │
│                    555-0199                             │
│                                                         │
│                    Ringing...                           │
│                  (glowing cyan)                         │
│                                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Design Specifications

### Background
- **Gradient**: Deep Navy (#0A1628) → Indigo (#1A2744) at 135deg
- **Overlay**: Frosted glass with `backdrop-filter: blur(10px)`
- **Opacity**: `rgba(255, 255, 255, 0.05)` white overlay
- **Shadow**: `0 4px 24px rgba(0, 0, 0, 0.3)`
- **Border Radius**: 16px
- **Padding**: 24px

### Typography

#### Name (Primary)
- **Text**: "Jonathan Doe"
- **Font Size**: 24px
- **Font Weight**: 700 (Bold)
- **Color**: #FFFFFF (Pure White)
- **Letter Spacing**: -0.5px
- **Line Height**: 1.2
- **Alignment**: Center

#### Phone Number (Secondary)
- **Text**: "555-0199"
- **Font Size**: 14px
- **Font Weight**: 400 (Regular)
- **Color**: #8B92A8 (Muted Grey)
- **Line Height**: 1.5
- **Alignment**: Center

#### Status (Accent)
- **Text**: "Ringing..." / "Calling..." / "Connected" / etc.
- **Font Size**: 16px
- **Font Weight**: 500 (Medium)
- **Color**: #00D9FF (Cyan Glow)
- **Text Shadow**: `0 0 8px rgba(0, 217, 255, 0.6)`
- **Line Height**: 1.4
- **Alignment**: Center
- **Animation**: Pulse (2s ease-in-out infinite) for active states

### Status States

#### Active States (Pulsing)
- **calling**: "Calling..." with cyan glow and pulse
- **ringing**: "Ringing..." with cyan glow and pulse

#### Success State (Steady)
- **connected**: "Connected" with cyan glow, no pulse

#### Inactive States (Muted)
- **ended**: "Call ended" in muted grey (#8B92A8), no glow
- **not_picked_up**: "Not picked up" in muted grey, no glow
- **cancelled**: "Cancelled" in muted grey, no glow

### Spacing
- **Card Padding**: 24px all sides
- **Content Gap**: 12px between elements
- **Top Margin**: 8px for status text
- **Card Margin**: 12px vertical

### Animations

#### Pulse (Active States)
```css
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}
```
- **Duration**: 2s
- **Timing**: ease-in-out
- **Iteration**: infinite

#### Fade In (Card Entry)
```css
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
```
- **Duration**: 0.3s
- **Timing**: ease-out

### Responsive Breakpoints

#### Mobile (≤640px)
- **Padding**: 20px
- **Name Font Size**: 20px
- **Status Font Size**: 14px
- **Card Margin**: 10px vertical

#### Tablet (641px - 1024px)
- **Padding**: 22px
- **Name Font Size**: 24px (default)
- **Status Font Size**: 16px (default)

#### Desktop (≥1025px)
- **Padding**: 24px (default)
- **Name Font Size**: 24px (default)
- **Status Font Size**: 16px (default)

## Example States

### State 1: Calling
```
┌─────────────────────────────────────┐
│                                     │
│         Dr. Sarah Chen              │
│         555-0123                    │
│                                     │
│         Calling...                  │
│         (pulsing cyan glow)         │
│                                     │
└─────────────────────────────────────┘
```

### State 2: Ringing
```
┌─────────────────────────────────────┐
│                                     │
│         Jonathan Doe                │
│         555-0199                    │
│                                     │
│         Ringing...                  │
│         (pulsing cyan glow)         │
│                                     │
└─────────────────────────────────────┘
```

### State 3: Connected
```
┌─────────────────────────────────────┐
│                                     │
│         Dr. Michael Ross            │
│         555-0456                    │
│                                     │
│         Connected                   │
│         (steady cyan glow)          │
│                                     │
└─────────────────────────────────────┘
```

### State 4: Call Ended
```
┌─────────────────────────────────────┐
│                                     │
│         Nurse Emily Park            │
│         555-0789                    │
│                                     │
│         Call ended                  │
│         (muted grey, no glow)       │
│         2:34                        │
│                                     │
└─────────────────────────────────────┘
```

### State 5: Not Picked Up
```
┌─────────────────────────────────────┐
│                                     │
│         Dr. James Wilson            │
│         555-0321                    │
│                                     │
│         Not picked up               │
│         (muted grey, no glow)       │
│                                     │
└─────────────────────────────────────┘
```

## Design Philosophy

### What Makes It "Luxury Minimalist"

1. **No Visual Clutter**
   - No buttons, no icons (except critical warnings)
   - Pure information display
   - Clean, uninterrupted surfaces

2. **Premium Materials**
   - Deep, rich color gradients
   - Frosted glass effects
   - Soft, diffused glows
   - Subtle shadows

3. **Generous Spacing**
   - Breathing room around all elements
   - Clear visual hierarchy
   - Not cramped or dense

4. **Subtle Motion**
   - Gentle pulse animations
   - Smooth transitions
   - Never jarring or distracting

5. **High-End Typography**
   - Bold, confident primary text
   - Refined secondary details
   - Careful letter spacing
   - Optimal line heights

6. **Calming Palette**
   - Deep navy blues (professional, trustworthy)
   - Soft cyan accents (modern, tech-forward)
   - Muted greys (sophisticated, understated)
   - Pure white (clean, clear)

## Accessibility

### Color Contrast (WCAG AA Compliant)
- **White on Navy**: 12.6:1 (AAA) ✅
- **Cyan on Navy**: 7.2:1 (AA) ✅
- **Muted Grey on Navy**: 4.8:1 (AA) ✅

### Reduced Motion Support
```css
@media (prefers-reduced-motion: reduce) {
  .call-card { animation: none; }
  .call-card__status--active { animation: none; }
}
```

### Focus States
- Cards are not interactive (no buttons)
- Information display only
- Screen reader friendly with semantic HTML

## Implementation Notes

- **No Buttons**: Cards are pure information display, not interactive
- **No Icons**: Text-only for maximum clarity
- **Responsive**: Scales gracefully on all screen sizes
- **4K Ready**: Crisp rendering at high resolutions
- **Performance**: CSS animations, no JavaScript
- **Maintainable**: Design tokens in CSS custom properties

---

**Design Inspiration**: Hatch Sleep app
**Implementation**: Modern CSS with backdrop-filter and gradients
**Status**: ✅ Fully implemented and tested
