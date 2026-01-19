# OpenAI Default Setup & Python Backend Removal

## Overview
Refactored the application to use OpenAI as the default AI provider and removed all Python backend dependencies. The app now exclusively uses the Vercel AI SDK with support for both OpenAI and Google Gemini.

## Changes Made

### 1. AIService Refactoring (`src/main/services/AIService.ts`)

**Multi-Provider Support:**
- Added support for both OpenAI and Google Gemini
- New methods:
  - `initializeOpenAI(apiKey, model)` - Initialize with OpenAI
  - `initializeGemini(apiKey, model)` - Initialize with Google Gemini
  - `setModel(model)` - Change active model
  - `getStatus()` - Get current provider, model, and initialization status
- Removed single `initialize()` method in favor of provider-specific methods
- Added `getModel()` private method to get the appropriate model instance

**Default Provider:**
- OpenAI is now the default provider
- Default model: `gpt-4o-mini` (fast and cost-effective)

### 2. Main Process Updates (`src/main/index.ts`)

**Removed Python Backend:**
- Removed `BACKEND_URL` constant
- Removed `conversationHistory` array (now managed by AIService)
- Removed Python backend health check in `backend:status` handler
- Removed Python backend fallback in `chat:send` handler

**API Key Loading:**
- Loads API keys from `api-config.json` file in userData directory on startup
- Falls back to environment variables if config file doesn't exist
- Automatically initializes the appropriate provider based on saved config
- Config file structure:
  ```json
  {
    "openaiApiKey": "sk-...",
    "geminiApiKey": "AIza...",
    "selectedModel": "gpt-4o-mini",
    "provider": "openai"
  }
  ```

**Updated IPC Handlers:**
- `backend:status` - Now returns AI service status instead of Python backend
- `chat:send` - Only uses AIService, throws error if not initialized
- `api:setKeys` - New unified handler for setting API keys (replaces `api:setGeminiKey`)
- `api:getStatus` - Returns provider, model, and initialization status

### 3. Preload API Updates (`src/preload/index.ts`)

**New API Methods:**
```typescript
api: {
  setKeys: (config: { 
    openaiApiKey?: string; 
    geminiApiKey?: string; 
    selectedModel?: string;
    provider?: 'openai' | 'gemini';
  }) => Promise<{ success: boolean; error?: string }>
  
  getStatus: () => Promise<{ 
    initialized: boolean; 
    provider: string; 
    currentModel: string 
  }>
}
```

### 4. Settings Modal Redesign (`src/renderer/components/SettingsModal.tsx`)

**Provider Selection:**
- Added visual provider selector with two buttons (OpenAI / Gemini)
- Shows provider-specific information and available models
- Automatically updates model options when provider changes

**Dynamic Model Selection:**
- Model dropdown changes based on selected provider
- OpenAI models: GPT-4o, GPT-4o Mini, GPT-4 Turbo
- Gemini models: 2.5 Flash Lite, 2.5 Flash, 2.5 Pro

**Improved UX:**
- Shows current provider and model in status banner
- Only shows API key input for selected provider
- Validates that selected provider has an API key before saving
- Saves provider preference along with API keys

**Default Values:**
- Provider: OpenAI
- Model: gpt-4o-mini

### 5. Configuration Persistence

**Storage Locations:**
1. **localStorage** (renderer process):
   - Key: `api_keys` - Contains all API keys and provider
   - Key: `selected_model` - Current model selection

2. **File System** (main process):
   - Path: `{userData}/api-config.json`
   - Contains same data as localStorage
   - Used for loading on app startup

**Auto-Initialization:**
- On app startup, main process loads config from file
- Automatically initializes the appropriate AI provider
- No manual configuration needed after first setup

## User Experience

### First Time Setup:
1. User opens app
2. Status shows "⚠ No API key configured"
3. User opens Settings → Models tab
4. OpenAI is pre-selected as default provider
5. User enters OpenAI API key
6. User clicks "Save Configuration"
7. AI service immediately initializes
8. User can start chatting

### Subsequent Launches:
1. App loads saved config from file
2. AI service auto-initializes with saved provider and model
3. User can immediately start chatting
4. No "AI Service not initialized" errors

## Error Handling

**No API Key Configured:**
- Error message: "AI Service not initialized. Please configure API keys in Settings."
- Shown in chat when user tries to send message without configuring
- Status banner in Settings shows warning

**Invalid API Key:**
- Error caught during chat request
- Displayed to user in chat interface
- User can update key in Settings

## Benefits

1. **No Python Dependency** - Removed entire Python backend, simplifying deployment
2. **Faster Startup** - No need to wait for Python process to start
3. **Better Error Messages** - Clear guidance to configure API keys
4. **Persistent Configuration** - API keys saved and auto-loaded
5. **Multi-Provider Support** - Easy to switch between OpenAI and Gemini
6. **OpenAI Default** - Best-in-class AI with reasonable pricing

## Migration Notes

**For Existing Users:**
- Old `.env` file with `GEMINI_API_KEY` will still work as fallback
- New users should configure via Settings UI
- Python backend code can be safely removed/ignored

**For Developers:**
- Python server (`python/server.py`) is no longer used
- Can remove Python dependencies if desired
- All AI functionality now through Vercel AI SDK

## Testing

To test the implementation:

1. **Fresh Install:**
   ```bash
   npm run dev
   ```
   - Should show "No API key configured" message
   - Open Settings → Models
   - Enter OpenAI API key
   - Save and verify chat works

2. **With Saved Config:**
   - Restart app after saving config
   - Should auto-initialize without errors
   - Chat should work immediately

3. **Provider Switching:**
   - Configure OpenAI key and test chat
   - Switch to Gemini provider
   - Enter Gemini key and save
   - Verify chat works with Gemini
   - Switch back to OpenAI
   - Verify chat works with OpenAI

## API Key Sources

- **OpenAI**: https://platform.openai.com/api-keys
- **Google Gemini**: https://aistudio.google.com/apikey
