# Offline Model Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Electron App                             │
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  Renderer Process │         │   Main Process   │             │
│  │  (React UI)       │         │   (Node.js)      │             │
│  │                   │         │                  │             │
│  │  ┌─────────────┐  │   IPC   │  ┌────────────┐ │             │
│  │  │SettingsModal│  │◄───────►│  │AIService   │ │             │
│  │  │  - Offline  │  │         │  │ - offline  │ │             │
│  │  │    Provider │  │         │  │   mode     │ │             │
│  │  └─────────────┘  │         │  └─────┬──────┘ │             │
│  │                   │         │        │        │             │
│  │  ┌─────────────┐  │         │  ┌─────▼──────┐ │             │
│  │  │  Chat UI    │  │◄───────►│  │LlamaService│ │             │
│  │  │  - Messages │  │         │  │ - spawn    │ │             │
│  │  │  - Input    │  │         │  │ - manage   │ │             │
│  │  └─────────────┘  │         │  │ - cleanup  │ │             │
│  │                   │         │  └─────┬──────┘ │             │
│  └───────────────────┘         │        │        │             │
│                                │        │        │             │
└────────────────────────────────┼────────┼────────┼─────────────┘
                                 │        │        │
                                 │   HTTP │        │ spawn/kill
                                 │        │        │
                        ┌────────▼────────▼────────▼──────────┐
                        │   llama-server (Port 8080)          │
                        │   ├─ llama.cpp runtime              │
                        │   ├─ Model: llama-3.2-3B-q4.gguf   │
                        │   ├─ Context: 2048 tokens           │
                        │   ├─ Threads: 4 CPU cores           │
                        │   └─ Local: 127.0.0.1 only          │
                        └─────────────────────────────────────┘
```

## Data Flow

### 1. User Sends Message

```
User types message
    │
    ▼
Chat UI (InputArea.tsx)
    │
    ▼
IPC: chat:send
    │
    ▼
AIService.chat() [Main Process]
    │
    ├─ Check provider === 'offline'
    │
    ▼
AIService.chatOffline()
    │
    ▼
LlamaService.chat()
    │
    ▼
HTTP POST to localhost:8080/v1/chat/completions
    │
    ▼
llama-server processes request
    │
    ├─ Load model from disk
    ├─ Process tokens
    ├─ Generate response
    │
    ▼
Response returned to LlamaService
    │
    ▼
Response sent back via IPC
    │
    ▼
Chat UI displays response
```

### 2. App Startup Sequence

```
App Launch (main/index.ts)
    │
    ▼
Load config from api-config.json
    │
    ├─ provider: 'offline'?
    │   │
    │   ▼ YES
    │   AIService.initializeOffline(modelPath)
    │       │
    │       ▼
    │   LlamaService.startServer()
    │       │
    │       ├─ Validate model file exists
    │       ├─ Validate llama-server exists
    │       ├─ spawn llama-server process
    │       ├─ Wait for server ready
    │       │   │
    │       │   ▼
    │       │   Poll http://localhost:8080/health
    │       │   │
    │       │   ▼ (max 30s)
    │       │   Server ready!
    │       │
    │       ▼
    │   Set provider = 'offline'
    │   Set model = 'llama3.2-3b-q4'
    │
    ▼
App ready for use
```

### 3. Settings Configuration Flow

```
User opens Settings → Models
    │
    ▼
Load current configuration
    │
    ├─ localStorage.getItem('api_keys')
    ├─ window.api.api.getStatus()
    └─ window.api.offline.getStatus()
    │
    ▼
Display current status
    │
    ├─ Provider: Offline Model
    ├─ Model Path: /path/to/model.gguf
    └─ Server Status: ✓ Running
    │
User changes provider to "Offline Model"
    │
    ▼
User enters model path
    │
    ▼
User clicks "Save Configuration"
    │
    ▼
IPC: api:setKeys({ 
    provider: 'offline', 
    offlineModelPath: '/path/to/model.gguf' 
})
    │
    ▼
Main Process:
    ├─ Save to api-config.json
    ├─ AIService.initializeOffline(modelPath)
    │   │
    │   ▼
    │   LlamaService.startServer(modelPath)
    │       │
    │       ├─ Stop existing server if running
    │       ├─ Start new server with new model
    │       └─ Wait for ready
    │
    ▼
Return success to renderer
    │
    ▼
UI shows "✓ Configuration saved successfully"
```

### 4. Cleanup on Exit

```
User closes app
    │
    ▼
app.on('will-quit')
    │
    ▼
llamaService.cleanup()
    │
    ▼
Kill llama-server process
    │
    ├─ serverProcess.kill('SIGTERM')
    ├─ Wait for graceful shutdown
    └─ Release resources
    │
    ▼
App exits cleanly
```

## Component Responsibilities

### LlamaService ([src/main/services/LlamaService.ts](src/main/services/LlamaService.ts))
- ✅ Spawn and manage llama-server process
- ✅ Configure server parameters (context, threads, batch size)
- ✅ Health checks and readiness detection
- ✅ HTTP communication with local server
- ✅ Process lifecycle management
- ✅ Error handling and recovery

### AIService ([src/main/services/AIService.ts](src/main/services/AIService.ts))
- ✅ Provider abstraction (OpenAI / Gemini / Offline)
- ✅ Chat message routing
- ✅ Conversation history management
- ✅ Provider switching logic
- ✅ Initialization and status tracking

### Main Process ([src/main/index.ts](src/main/index.ts))
- ✅ IPC handler registration
- ✅ Configuration persistence
- ✅ Auto-start offline model on launch
- ✅ Cleanup on app exit

### Settings UI ([src/renderer/components/SettingsModal.tsx](src/renderer/components/SettingsModal.tsx))
- ✅ Provider selection UI
- ✅ Model path configuration
- ✅ Server status display
- ✅ Save/load configuration
- ✅ Real-time status updates

### Preload Script ([src/preload/index.ts](src/preload/index.ts))
- ✅ Secure IPC bridge
- ✅ API exposure to renderer
- ✅ Type-safe communication

## Security & Privacy

### Privacy Guarantees
✅ All data stays on device
✅ No external API calls
✅ No telemetry or tracking
✅ Conversation history never leaves machine

### Security Measures
✅ Server binds to localhost only (127.0.0.1)
✅ No network exposure
✅ Process isolation via Electron
✅ Sandboxed renderer process
✅ Context isolation enabled
✅ No direct Node.js access from UI

## Performance Characteristics

### Resource Usage
- **RAM**: 2-3 GB (model loaded in memory)
- **CPU**: 4 threads by default (configurable)
- **Disk**: ~2 GB for Q4 model file
- **Network**: None (localhost only)

### Response Times
- **First token**: 1-3 seconds (cold start)
- **Subsequent tokens**: 50-100ms each (CPU)
- **With GPU**: 10-30ms per token

### Scalability
- **Context window**: 2048 tokens (configurable)
- **Concurrent requests**: Single-threaded (sequential)
- **Model switching**: Requires server restart

## Configuration Options

### Server Parameters (in LlamaService.ts)
```typescript
{
  ctxSize: 2048,        // Context window
  threads: 4,           // CPU threads
  batchSize: 256,       // Batch processing size
  port: 8080,           // Server port
  gpuLayers: 0,         // GPU offloading (0=CPU only)
}
```

### Model Specifications
- **Format**: GGUF (required)
- **Quantization**: Q4_K_M (recommended)
- **Size**: ~2GB (3B parameters)
- **Type**: Llama 3.2 architecture

## Future Enhancements

### Potential Improvements
- [ ] Multiple model support (model switching)
- [ ] GPU acceleration by default
- [ ] Streaming responses (token-by-token)
- [ ] Tool calling support for offline models
- [ ] Model downloading/management UI
- [ ] Performance monitoring dashboard
- [ ] Context window adjustment based on available RAM
- [ ] Batch processing for multiple messages

### Advanced Features
- [ ] Fine-tuning support
- [ ] Custom prompt templates
- [ ] Response caching
- [ ] Multi-model ensemble
- [ ] Quantization options (Q2, Q4, Q8)
