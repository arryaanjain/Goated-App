# Quick Setup Steps - Offline Model Integration

## 🚀 Quick Start (5 Minutes)

### Step 1: Prepare Your Model

```bash
# Navigate to your project
cd /home/arryaanjain/Desktop/Everything/Goated-App

# Create models directory
mkdir -p models

# Copy your Llama 3.2 3B Q4 GGUF model
# Replace SOURCE_PATH with your actual model location
cp /path/to/llama-3.2-3B-q4_K_M.gguf ./models/
```

### Step 2: Set Up llama.cpp

If you already have llama.cpp built:
```bash
# Create symlink to your llama.cpp directory
ln -s /path/to/your/llama.cpp ./llama.cpp
```

OR if you need to build it:
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
cd ..
```

Verify it's working:
```bash
ls -la llama.cpp/build/bin/llama-server
# Should show the executable
```

### Step 3: Verify File Structure

```bash
# Your project should look like this:
tree -L 3 -I node_modules

Goated-App/
├── llama.cpp/              # llama.cpp repo or symlink
│   └── build/
│       └── bin/
│           └── llama-server  ✅
├── models/                  # Your models directory
│   └── llama-3.2-3B-q4_K_M.gguf  ✅
├── src/
│   ├── main/
│   │   └── services/
│   │       ├── AIService.ts
│   │       └── LlamaService.ts  ✅ NEW
│   └── renderer/
│       └── components/
│           └── SettingsModal.tsx  ✅ UPDATED
└── package.json
```

### Step 4: Build and Run

```bash
# Install dependencies (if not already done)
npm install

# Build the app
npm run build

# Run in development mode
npm run dev
```

### Step 5: Verify It's Working

1. **App launches** ✅
2. **Check console** for:
   ```
   [LlamaService] Starting llama-server...
   [LlamaService] Server is ready!
   [AIService] Offline model initialized successfully
   ```
3. **Open Settings** → Models tab
4. **Verify status**: Should show "✓ Server Running"
5. **Send a test message** in chat: "Hello!"

## 📋 Implementation Summary

### Files Created:
1. ✅ [`src/main/services/LlamaService.ts`](src/main/services/LlamaService.ts) - Manages llama-server process
2. ✅ [`OFFLINE-MODEL-SETUP.md`](OFFLINE-MODEL-SETUP.md) - Detailed setup guide

### Files Modified:
1. ✅ [`src/main/services/AIService.ts`](src/main/services/AIService.ts)
   - Added `initializeOffline()` method
   - Added `chatOffline()` for local model chat
   - Changed default provider to `'offline'`
   - Updated `isInitialized()` to check llama server status

2. ✅ [`src/main/index.ts`](src/main/index.ts)
   - Imported `llamaService`
   - Auto-start offline model on app launch
   - Added cleanup on app quit
   - Added IPC handlers: `offline:startServer`, `offline:stopServer`, `offline:getStatus`
   - Updated `api:setKeys` to support offline provider

3. ✅ [`src/preload/index.ts`](src/preload/index.ts)
   - Added `offline` namespace to API
   - Exposed offline model control methods to renderer

4. ✅ [`src/renderer/components/SettingsModal.tsx`](src/renderer/components/SettingsModal.tsx)
   - Added "Offline Model" provider option
   - Added offline model path configuration UI
   - Added server status display
   - Changed default provider to `'offline'`
   - Updated save logic to handle offline model initialization

## 🔧 Configuration Details

### Server Parameters
The llama-server runs with:
- **Context Size**: 2048 tokens
- **Threads**: 4 CPU threads
- **Batch Size**: 256
- **Port**: 8080 (localhost only)
- **GPU Layers**: 0 (CPU-only by default)

### Default Paths
- **Model**: `./models/llama-3.2-3B-q4_K_M.gguf`
- **Server**: `./llama.cpp/build/bin/llama-server`
- **Config**: `~/.config/GoatedApp/api-config.json` (auto-saved)

## 🎯 Key Features

✅ **Privacy-First**: All processing happens locally
✅ **Auto-Start**: Model starts automatically on app launch
✅ **No API Keys**: No external services needed
✅ **Offline**: Works completely offline
✅ **UI Integration**: Full settings UI for configuration
✅ **Status Monitoring**: Real-time server status display
✅ **Error Handling**: Comprehensive error messages
✅ **Graceful Shutdown**: Proper cleanup on app exit

## 🔍 Testing Checklist

- [ ] Model file exists at `models/llama-3.2-3B-q4_K_M.gguf`
- [ ] llama-server executable exists and is runnable
- [ ] App starts without errors
- [ ] Console shows "Server is ready!"
- [ ] Settings shows "✓ Server Running"
- [ ] Can send messages and get responses
- [ ] No external network calls (check DevTools Network tab)
- [ ] App closes cleanly (server stops)

## 🚨 Common Issues & Quick Fixes

### Issue: Model not found
```bash
# Check file exists
ls -la models/llama-3.2-3B-q4_K_M.gguf

# If not, copy it:
cp /path/to/your/model.gguf ./models/llama-3.2-3B-q4_K_M.gguf
```

### Issue: llama-server not found
```bash
# Check it exists
ls -la llama.cpp/build/bin/llama-server

# If not, build it:
cd llama.cpp && make && cd ..
```

### Issue: Server won't start
```bash
# Check if port 8080 is available
lsof -i :8080

# If occupied, kill it or change port in LlamaService.ts
```

### Issue: Slow responses
- Reduce context size to 1024
- Increase threads to match your CPU cores
- Consider GPU acceleration

## 📚 Documentation

- **Full Setup Guide**: [OFFLINE-MODEL-SETUP.md](OFFLINE-MODEL-SETUP.md)
- **llama.cpp Docs**: https://github.com/ggerganov/llama.cpp
- **Llama 3.2 Model**: https://huggingface.co/meta-llama/Llama-3.2-3B

## 🎉 Success!

Your app now uses a fully offline AI model by default. No API keys, no external services, complete privacy!

**Next Steps:**
1. Test thoroughly with various prompts
2. Adjust server parameters if needed for performance
3. Consider GPU acceleration for faster responses
4. Add more models for different use cases

---

**Default Behavior**: The app will automatically try to start the offline model from `./models/llama-3.2-3B-q4_K_M.gguf` on launch. If not found, you can configure it manually in Settings → Models → Offline Model.
