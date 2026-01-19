# Card Persistence Fix

## Issue

Cards were displaying during streaming but **disappearing after the message was saved** to the database and the page was refreshed.

### Symptoms
1. Card renders correctly during streaming ✅
2. Card disappears after stream completes ❌
3. Card doesn't show after page refresh ❌
4. Only text message remains visible

## Root Cause

When converting database messages to UI format in `App.tsx`, the **`result` field was missing** from the toolCalls mapping.

### Code Issue

**BEFORE (Broken):**
```typescript
toolCalls: msg.toolCalls?.map(tc => ({
  id: tc.id,
  name: tc.name,
  arguments: tc.arguments,
  status: tc.status
  // ❌ Missing: result field
})),
```

**AFTER (Fixed):**
```typescript
toolCalls: msg.toolCalls?.map(tc => ({
  id: tc.id,
  name: tc.name,
  arguments: tc.arguments,
  result: tc.result, // ✅ IMPORTANT: Include result for card rendering
  status: tc.status
})),
```

## Why This Matters

The `result` field contains the **actual JSON response** from the MCP tool:

```json
{
  "success": true,
  "data": {
    "staff": [
      { "name": "Dr. Mohan", "role": "doctor", ... },
      { "name": "Jahid", "role": "administrator", ... }
    ],
    "count": 8
  }
}
```

Without the `result` field:
- CardRenderer receives empty string `''`
- Cannot parse tool output
- Falls back to ToolCallCard (or shows nothing)
- Card data is lost

With the `result` field:
- CardRenderer receives full JSON response
- Parses tool output successfully
- Renders appropriate card (StaffListCard, CallCard, etc.)
- Card persists across page refreshes ✅

## Flow Comparison

### During Streaming (Was Working)
```
Tool Executes
  ↓
streamingToolCalls includes result
  ↓
CardRenderer receives result
  ↓
Card renders ✅
```

### After Database Save (Was Broken)
```
Message saved to DB with result
  ↓
Message loaded from DB
  ↓
toolCalls mapped WITHOUT result ❌
  ↓
CardRenderer receives empty string
  ↓
Card doesn't render ❌
```

### After Fix (Now Working)
```
Message saved to DB with result
  ↓
Message loaded from DB
  ↓
toolCalls mapped WITH result ✅
  ↓
CardRenderer receives result
  ↓
Card renders ✅
```

## Database Schema

The database already stores the `result` field correctly:

```typescript
// src/renderer/db/schemas.ts
export interface ToolCallData {
  id: string;
  name: string;
  arguments: string;
  result?: string; // ✅ Stored in database
  status: 'pending' | 'success' | 'error';
}
```

The issue was only in the **UI mapping layer** in `App.tsx`.

## Testing

### Test Steps
1. Ask: "Show me all staff members"
2. Wait for StaffListCard to render during streaming
3. Wait for stream to complete and message to save
4. Verify card remains visible ✅
5. Refresh the page (Cmd+R / Ctrl+R)
6. Verify card is still visible ✅

### Expected Behavior
- ✅ Card renders during streaming
- ✅ Card persists after stream completes
- ✅ Card persists after page refresh
- ✅ Card data is complete and accurate
- ✅ Multiple cards in same message all persist

## Files Changed

**`src/renderer/App.tsx`** (Line 86)
- Added `result: tc.result` to toolCalls mapping
- Added comment explaining importance

## Related Issues

This is the **same issue** that was fixed before in `CARD-RENDERING-FIX.md`, but it appears the fix was not applied or was reverted. This fix ensures:

1. Cards persist across page refreshes
2. Cards show correct data from database
3. CardRenderer can parse tool output
4. All card types work correctly (CallCard, StaffListCard, etc.)

## Prevention

To prevent this issue in the future:

1. **Always include `result` field** when mapping toolCalls
2. **Test persistence** by refreshing page after card renders
3. **Check database** to ensure result is being saved
4. **Verify UI mapping** includes all necessary fields

## Summary

**Problem**: Cards disappeared after streaming because `result` field was missing from database-to-UI mapping.

**Solution**: Added `result: tc.result` to the toolCalls mapping in `App.tsx` line 86.

**Result**: Cards now persist correctly across page refreshes and show complete data from the database.

---

**Fix Date**: January 19, 2026  
**Status**: ✅ Complete  
**Impact**: All card types now persist correctly  
