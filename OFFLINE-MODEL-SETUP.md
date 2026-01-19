# Offline Model Setup Guide

This guide will help you set up and configure the offline Llama 3.2 3B model with your Electron app.

## Prerequisites

✅ **Already Complete:**
- llama.cpp built with `llama-server` available in `build/bin`
- Llama 3.2 3B model quantized to Q4 (GGUF format)

## Directory Structure Setup

### 1. Create Models Directory

Create a `models` folder in your project root to store your GGUF model file:

```bash
cd /home/arryaanjain/Desktop/Everything/Goated-App
mkdir -p models
```

### 2. Place Your Model File

Move your Llama 3.2 3B Q4 GGUF file into the models directory:

```bash
# Example - adjust the source path to where your model is located
cp /path/to/your/llama-3.2-3B-q4_K_M.gguf ./models/
```

**Expected file location:** `/home/arryaanjain/Desktop/Everything/Goated-App/models/llama-3.2-3B-q4_K_M.gguf`

### 3. Set Up llama.cpp

Create a symlink or directory for llama.cpp in your project root:

```bash
# Option 1: Symlink to your existing llama.cpp build
ln -s /path/to/your/llama.cpp ./llama.cpp

# Option 2: If you haven't built llama.cpp yet
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
# This will create build/bin/llama-server
```

**Required structure:**
```
Goated-App/
├── llama.cpp/
│   └── build/
│       └── bin/
│           └── llama-server  (executable)
└── models/
    └── llama-3.2-3B-q4_K_M.gguf
```

## Configuration

### Method 1: Auto-Start (Default)

The app will automatically try to start the offline model from the default path on launch:

**Default path:** `./models/llama-3.2-3B-q4_K_M.gguf`

If your model is at this location, it will start automatically.

### Method 2: Manual Configuration via Settings

1. **Launch the app:**
   ```bash
   npm run dev
   ```

2. **Open Settings:**
   - Click the Settings/gear icon
   - Navigate to the "Models" tab

3. **Configure Offline Model:**
   - Select "🔒 Offline Model" as your provider
   - Enter the full path to your GGUF file:
     ```
     /home/arryaanjain/Desktop/Everything/Goated-App/models/llama-3.2-3B-q4_K_M.gguf
     ```
   - Click "Save Configuration"

4. **Verify Status:**
   - You should see "✓ Server Running" in the status section
   - The model will be ready to use in the chat

## Server Configuration

The llama-server is configured with the following parameters:

```bash
--ctx-size 2048       # Context window size
--threads 4           # CPU threads to use
--batch-size 256      # Batch size for processing
--port 8080           # Server port (localhost only)
--host 127.0.0.1      # Bind to localhost only
--n-gpu-layers 0      # CPU only (change if you want GPU)
```

### GPU Support (Optional)

If you have a CUDA-capable GPU and want to use it:

1. Build llama.cpp with CUDA support:
   ```bash
   cd llama.cpp
   make clean
   make LLAMA_CUDA=1
   ```

2. Edit [LlamaService.ts](src/main/services/LlamaService.ts#L45) and change:
   ```typescript
   '--n-gpu-layers', '0',  // Change to '33' or higher
   ```

## Testing the Integration

### 1. Check Server Status

Open the app and check the console (DevTools):

```
[LlamaService] Starting llama-server...
[LlamaService] Model: /path/to/model.gguf
[LlamaService] Server is ready!
[AIService] Offline model initialized successfully
```

### 2. Send a Test Message

Try sending a message in the chat:
- Type: "Hello, how are you?"
- Expected: Response from the local model
- The response will appear without any API calls to external services

### 3. Verify Privacy

Check your network traffic - there should be NO outbound connections except to `localhost:8080`

## Troubleshooting

### Model File Not Found

**Error:** `Model file not found: /path/to/model.gguf`

**Solution:**
- Verify the file path is correct
- Check file permissions: `ls -la models/`
- Ensure the file has `.gguf` extension

### llama-server Not Found

**Error:** `llama-server not found at: /path/to/llama.cpp/build/bin/llama-server`

**Solution:**
- Verify llama.cpp is built: `ls -la llama.cpp/build/bin/`
- Rebuild if necessary: `cd llama.cpp && make`
- Check the path in [LlamaService.ts](src/main/services/LlamaService.ts#L29):
  ```typescript
  const llamaServerPath = path.join(process.cwd(), 'llama.cpp', 'build', 'bin', 'llama-server');
  ```

### Server Startup Timeout

**Error:** `Server startup timeout`

**Solution:**
- Increase timeout in [LlamaService.ts](src/main/services/LlamaService.ts#L74)
- Check system resources (RAM, CPU)
- Try a smaller model or lower context size

### Slow Responses

**Performance Tips:**
1. Reduce context size: `--ctx-size 1024`
2. Increase threads: `--threads 8` (based on your CPU)
3. Use GPU acceleration (see GPU Support section)
4. Use a smaller quantization (Q4 vs Q8)

### Memory Issues

**Out of Memory:**
- Q4 3B model needs ~2-3GB RAM
- Close other applications
- Reduce `--ctx-size` to 1024 or 512
- Consider using a smaller model

## Performance Benchmarks

**Expected Performance (3B Q4 on CPU):**
- First token: 1-3 seconds
- Subsequent tokens: 50-100ms each
- Context window: 2048 tokens
- Memory usage: 2-3GB RAM

**With GPU (CUDA):**
- First token: 200-500ms
- Subsequent tokens: 10-30ms each
- Significant speedup for longer contexts

## File Locations Reference

| Component | Path |
|-----------|------|
| LlamaService | [src/main/services/LlamaService.ts](src/main/services/LlamaService.ts) |
| AIService | [src/main/services/AIService.ts](src/main/services/AIService.ts) |
| IPC Handlers | [src/main/index.ts](src/main/index.ts) |
| Settings UI | [src/renderer/components/SettingsModal.tsx](src/renderer/components/SettingsModal.tsx) |
| Preload API | [src/preload/index.ts](src/preload/index.ts) |
| Model File | `models/llama-3.2-3B-q4_K_M.gguf` |
| Server Binary | `llama.cpp/build/bin/llama-server` |

## Privacy & Security

✅ **Privacy Features:**
- All data stays on your device
- No external API calls
- Model runs 100% locally
- Conversation history never leaves your machine

✅ **Security:**
- Server binds to localhost only (127.0.0.1)
- No network exposure
- Process isolation via Electron

## Next Steps

1. ✅ Place your GGUF model in `models/` directory
2. ✅ Verify llama-server is at `llama.cpp/build/bin/llama-server`
3. ✅ Run `npm run dev` to start the app
4. ✅ Configure in Settings or use auto-start
5. ✅ Test with a simple message
6. ✅ Enjoy private, offline AI conversations!

## Support

If you encounter issues:
1. Check the console logs in DevTools (Ctrl+Shift+I)
2. Review this troubleshooting section
3. Check llama.cpp documentation: https://github.com/ggerganov/llama.cpp
4. Verify model compatibility (GGUF format required)

## Advanced Configuration

### Custom Server Parameters

Edit [LlamaService.ts](src/main/services/LlamaService.ts#L37-L45) to customize:

```typescript
this.serverProcess = spawn(llamaServerPath, [
  '--model', modelPath,
  '--ctx-size', '4096',      // Increase context
  '--threads', '8',          // More CPU threads
  '--batch-size', '512',     // Larger batches
  '--port', this.serverPort.toString(),
  '--host', '127.0.0.1',
  '--n-gpu-layers', '33',    // Use GPU
  '--rope-freq-base', '500000', // Extended context
]);
```

### Multiple Models

To support multiple models, update the UI in [SettingsModal.tsx](src/renderer/components/SettingsModal.tsx) to add model selection:

```typescript
const offlineModels = [
  { value: 'llama3.2-3b-q4', path: 'models/llama-3.2-3B-q4_K_M.gguf' },
  { value: 'llama3.2-1b-q4', path: 'models/llama-3.2-1B-q4_K_M.gguf' },
];
```

---

**Note:** This is set as the default provider. When you first launch the app, it will try to auto-start the offline model if found at the default path.
