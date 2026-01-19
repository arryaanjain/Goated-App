# Packaging Guide - Bundle Offline Model with App

## Overview

The app now **automatically detects** the bundled offline model - no path configuration needed! The model will be packaged with your executable.

## Directory Structure

### Development Mode
```
Goated-App/
├── models/
│   └── llama-3.2-3B-q4_K_M.gguf  ← Place model here
├── llama.cpp/
│   └── build/bin/
│       └── llama-server
└── src/
```

### Production (Packaged App)
```
GoatedApp.AppImage (or .exe, .dmg)
└── resources/
    ├── models/
    │   └── llama-3.2-3B-q4_K_M.gguf  ← Auto-bundled
    ├── bin/
    │   └── llama-server  ← Auto-bundled
    └── app.asar
```

## Setup for Packaging

### 1. Place Model in Project

```bash
cd /home/arryaanjain/Desktop/Everything/Goated-App

# Create models directory
mkdir -p models

# Copy your model
cp /path/to/llama-3.2-3B-q4_K_M.gguf ./models/
```

### 2. Place llama-server Binary

```bash
# Create bin directory
mkdir -p bin

# Copy llama-server executable
cp llama.cpp/build/bin/llama-server ./bin/

# Make executable
chmod +x bin/llama-server
```

### 3. Update package.json

Add Electron Builder configuration to include the model and binary:

```json
{
  "name": "goated-app",
  "version": "1.0.0",
  "build": {
    "appId": "com.goatedapp.app",
    "productName": "GoatedApp",
    "files": [
      "dist/**/*",
      "node_modules/**/*",
      "package.json"
    ],
    "extraResources": [
      {
        "from": "models",
        "to": "models",
        "filter": ["**/*.gguf"]
      },
      {
        "from": "bin",
        "to": "bin",
        "filter": ["llama-server"]
      }
    ],
    "linux": {
      "target": ["AppImage", "deb"],
      "category": "Utility"
    },
    "win": {
      "target": ["nsis", "portable"]
    },
    "mac": {
      "target": ["dmg", "zip"],
      "category": "public.app-category.productivity"
    }
  }
}
```

### 4. Install Electron Builder (if not already)

```bash
npm install --save-dev electron-builder
```

### 5. Add Build Scripts

Update `package.json` scripts:

```json
{
  "scripts": {
    "dev": "...",
    "build": "tsc && vite build",
    "package": "npm run build && electron-builder",
    "package:linux": "npm run build && electron-builder --linux",
    "package:win": "npm run build && electron-builder --win",
    "package:mac": "npm run build && electron-builder --mac"
  }
}
```

## How Auto-Detection Works

The app automatically finds the model based on environment:

```typescript
// Development
model: /home/user/Goated-App/models/llama-3.2-3B-q4_K_M.gguf

// Production
model: /opt/GoatedApp/resources/models/llama-3.2-3B-q4_K_M.gguf
```

No user configuration needed!

## Building the Package

### Linux (AppImage)

```bash
npm run package:linux
```

**Output**: `dist/GoatedApp-1.0.0.AppImage` (~2.5 GB with model)

### Windows (Installer)

```bash
npm run package:win
```

**Output**: `dist/GoatedApp Setup 1.0.0.exe` (~2.5 GB with model)

### macOS (DMG)

```bash
npm run package:mac
```

**Output**: `dist/GoatedApp-1.0.0.dmg` (~2.5 GB with model)

## What Gets Bundled

| Item | Size | Purpose |
|------|------|---------|
| App code | ~50 MB | Your Electron app |
| Node modules | ~200 MB | Dependencies |
| Model file | ~2 GB | Llama 3.2 3B Q4 |
| llama-server | ~50 MB | Inference engine |
| **Total** | **~2.3 GB** | Complete package |

## Testing the Package

### 1. Build Development Package

```bash
npm run build
npm run package:linux
```

### 2. Install and Run

```bash
# Make executable
chmod +x dist/GoatedApp-1.0.0.AppImage

# Run
./dist/GoatedApp-1.0.0.AppImage
```

### 3. Verify Auto-Detection

Check the console (if you can access it):
```
[LlamaService] Starting llama-server...
[LlamaService] Model: /tmp/.mount_GoatedXXXXXX/resources/models/llama-3.2-3B-q4_K_M.gguf
[LlamaService] Server is ready!
```

Or check Settings → Models:
- Should show "✓ Running"
- Should display model path automatically

## Distribution

### Linux
- **AppImage**: Single file, portable, no installation
- **DEB**: For Debian/Ubuntu systems

### Windows
- **NSIS Installer**: Traditional Windows installer
- **Portable**: Standalone .exe, no installation

### macOS
- **DMG**: Disk image for drag-to-Applications
- **ZIP**: Archive for direct extraction

## File Sizes

**Uncompressed Package**: ~2.3 GB
**Compressed (zip)**: ~1.5 GB (depending on model quantization)

To reduce size:
- Use Q2 quantization instead of Q4 (~1 GB model)
- Use smaller model (1B instead of 3B)
- Offer model as separate download

## Optional: Separate Model Download

If package size is too large, you can:

1. **Ship without model initially**
2. **Download model on first run**
3. **Show progress bar during download**

This would reduce initial download from 2.3 GB to ~300 MB.

See `OPTIONAL-MODEL-DOWNLOAD.md` for implementation.

## Platform-Specific Notes

### Linux
- Model bundled in AppImage works automatically
- No permission issues
- Portable, single file

### Windows
- Ensure llama-server.exe (not just llama-server)
- May need Visual C++ Redistributable
- Antivirus might flag the executable

### macOS
- Code signing required for distribution
- Notarization needed for Gatekeeper
- Apple Silicon vs Intel builds

## Verification Checklist

Before distributing:

- [ ] Model file in `models/` directory
- [ ] llama-server in `bin/` directory
- [ ] `package.json` has `extraResources` config
- [ ] Build completes without errors
- [ ] Packaged app runs and detects model
- [ ] Settings shows "✓ Running"
- [ ] Chat works without API keys
- [ ] No external network calls
- [ ] App closes cleanly

## Common Issues

### Issue: Model not found in package

**Cause**: Not included in extraResources

**Fix**: Verify `package.json` has:
```json
"extraResources": [
  { "from": "models", "to": "models" }
]
```

### Issue: llama-server not executable

**Cause**: Permissions not preserved

**Fix**: Set executable bit before packaging:
```bash
chmod +x bin/llama-server
```

### Issue: Package too large

**Solutions**:
1. Use Q2 quantization (~1 GB)
2. Use 1B model instead of 3B
3. Separate model download
4. Offer "lite" and "full" versions

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Download Model
        run: |
          mkdir -p models
          wget https://huggingface.co/.../llama-3.2-3B-q4.gguf -O models/llama-3.2-3B-q4_K_M.gguf
      
      - name: Download llama-server
        run: |
          # Download pre-built llama-server or build from source
          mkdir -p bin
          # ... download/build steps
      
      - name: Install Dependencies
        run: npm ci
      
      - name: Build and Package
        run: npm run package:linux
      
      - name: Upload Artifacts
        uses: actions/upload-artifact@v3
        with:
          name: linux-appimage
          path: dist/*.AppImage
```

## Summary

✅ **Model is auto-detected** - no user configuration needed
✅ **Bundled with app** - works out of the box
✅ **Fully offline** - no downloads on first run
✅ **Platform-specific** - optimized for each OS

Your users just download, install, and use - the offline model works immediately!
