# Quick Start Guide - OpenAI Setup

## Fixed Issues ✅

1. **"AI Service not initialized" Error** - FIXED
   - App now auto-loads API keys from saved configuration
   - Clear error messages guide you to Settings if not configured

2. **Python Backend Removed** - DONE
   - No more Python backend dependency
   - Exclusively uses OpenAI/Gemini via Vercel AI SDK
   - Faster startup, simpler deployment

3. **OpenAI as Default** - DONE
   - OpenAI is now the default provider
   - GPT-4o Mini is the default model (fast & affordable)

## First Time Setup (5 minutes)

### Step 1: Get an OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

### Step 2: Configure the App
1. Run the app: `npm run dev`
2. Click the Settings icon (⚙️) in the bottom left
3. Click the "Models" tab
4. OpenAI should already be selected
5. Paste your API key in the "OpenAI API Key" field
6. Click "Save Configuration"

### Step 3: Start Chatting
1. Close the Settings modal
2. Type a message in the chat input
3. Press Enter or click Send
4. The AI will respond using GPT-4o Mini

## That's It! 🎉

Your app is now configured and ready to use. The API key is saved locally and will be automatically loaded on future app launches.

## Optional: Switch to Gemini

If you prefer Google Gemini:

1. Get a Gemini API key from https://aistudio.google.com/apikey
2. Open Settings → Models
3. Click the "Google Gemini" button
4. Enter your Gemini API key
5. Select a Gemini model
6. Click "Save Configuration"

## Troubleshooting

### "AI Service not initialized" Error
- Open Settings → Models
- Check that you have an API key entered
- Make sure the correct provider is selected
- Click "Save Configuration"

### Chat Not Working
- Check your API key is valid
- Verify you have credits/quota on your OpenAI account
- Check the browser console for error messages

### Want to Change Models?
- Open Settings → Models
- Select a different model from the dropdown
- Click "Save Configuration"
- New model will be used for all future chats

## Model Recommendations

**For Speed & Cost:**
- GPT-4o Mini (OpenAI) - Fast, cheap, good quality

**For Best Quality:**
- GPT-4o (OpenAI) - Best reasoning and accuracy
- Gemini 2.5 Pro (Google) - Excellent for complex tasks

**For Balance:**
- GPT-4 Turbo (OpenAI) - Good balance of speed and quality
- Gemini 2.5 Flash (Google) - Fast with good quality

## Cost Estimates (OpenAI)

- **GPT-4o Mini**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- **GPT-4o**: ~$2.50 per 1M input tokens, ~$10 per 1M output tokens

For typical chat usage (100 messages/day), expect:
- GPT-4o Mini: < $1/month
- GPT-4o: ~$5-10/month

## Support

If you encounter issues:
1. Check the console logs (View → Toggle Developer Tools)
2. Verify your API key is correct
3. Check your OpenAI account has available credits
4. Try restarting the app
