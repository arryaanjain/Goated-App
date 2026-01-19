# Visual Setup Guide - Step by Step

## 🎯 Goal
Get your Electron app running with Llama 3.2 3B offline model in 5 minutes.

---

## Step 1️⃣: Prepare Directory Structure

```bash
cd /home/arryaanjain/Desktop/Everything/Goated-App
mkdir -p models
```

---

## Step 2️⃣: Copy Your Model File

```bash
# Replace SOURCE_PATH with your actual path
cp /path/to/your/llama-3.2-3B-q4_K_M.gguf ./models/

# Verify (should show ~2GB file)
ls -lh models/
```

---

## Step 3️⃣: Set Up llama.cpp

**Link existing build:**
```bash
ln -s /path/to/your/llama.cpp ./llama.cpp
```

**OR build fresh:**
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make && cd ..
```

**Make executable:**
```bash
chmod +x llama.cpp/build/bin/llama-server
```

---

## Step 4️⃣: Verify Setup

```bash
# All three should show ✅
[ -f models/llama-3.2-3B-q4_K_M.gguf ] && echo "✅ Model OK" || echo "❌ Model MISSING"
[ -x llama.cpp/build/bin/llama-server ] && echo "✅ Server OK" || echo "❌ Server MISSING"
[ -f src/main/services/LlamaService.ts ] && echo "✅ Code OK" || echo "❌ Code MISSING"
```

---

## Step 5️⃣: Build & Run

```bash
npm install
npm run build
npm run dev
```

**Look for in console:**
```
[LlamaService] Server is ready!  ← ✅
[AIService] Offline model initialized successfully  ← ✅
```

---

## Step 6️⃣: Test

1. **Open Settings → Models tab**
2. **Verify**: "Server Status: ✓ Running"
3. **Test chat**: Type "Hello!" and press Enter
4. **Expect**: Response in 1-3 seconds

---

## ✅ Success Indicators

- Console: "Server is ready!"
- Settings: "✓ Server Running"
- Chat: Gets responses
- Network tab: Only localhost calls

---

## 🚨 Troubleshooting

**Model not found?**
```bash
cp /path/to/model.gguf ./models/llama-3.2-3B-q4_K_M.gguf
```

**Server not found?**
```bash
cd llama.cpp && make && cd ..
chmod +x llama.cpp/build/bin/llama-server
```

**Port conflict?**
```bash
lsof -i :8080  # Find what's using port
# Edit LlamaService.ts to change port
```

---

**Done! Your app now runs 100% offline with complete privacy!** 🎉
