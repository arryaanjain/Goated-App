# Fix API Key Leak - Step by Step Instructions

## ⚠️ IMPORTANT: Revoke the Exposed API Key First!

**Before doing anything else:**
1. Go to https://platform.openai.com/api-keys
2. Find the key starting with `sk-proj-fCpw8BAVZgsYYsZj9a_5oW4gQ0t3DNDye0v9UZVynigSULowOO...`
3. **REVOKE IT IMMEDIATELY**
4. Generate a new API key
5. Save the new key to `python/.env` file

## Option 1: Simple Fix (Recommended)

Since the API key is only in one recent commit, the easiest solution is:

```bash
# 1. Create a backup branch (just in case)
git branch backup-before-fix

# 2. Reset to the commit before the API key was added
git reset --hard 5d713cc

# 3. Re-apply your changes (the file is already fixed)
git add .
git commit -m "Add model configuration and MCP persistence (API key removed)"

# 4. Force push to GitHub
git push origin main --force
```

## Option 2: Using Git Filter-Branch (More thorough)

This removes the file from ALL history:

```bash
# 1. Create backup
git branch backup-before-fix

# 2. Remove the file from all commits
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch python/openai-test.py' \
  --prune-empty --tag-name-filter cat -- --all

# 3. Clean up
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 4. Re-add the fixed file
git add python/openai-test.py
git commit -m "Add openai-test.py without hardcoded API key"

# 5. Force push
git push origin main --force
```

## Option 3: Using BFG Repo-Cleaner (Fastest)

```bash
# 1. Install BFG (if not installed)
brew install bfg  # macOS
# or download from: https://rtyley.github.io/bfg-repo-cleaner/

# 2. Create backup
git branch backup-before-fix

# 3. Run BFG to remove the API key
bfg --replace-text <(echo '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==>***REMOVED***')

# 4. Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. Force push
git push origin main --force
```

## After Fixing

1. ✅ Verify the API key is gone:
   ```bash
   git log -p python/openai-test.py | grep "sk-proj"
   ```
   Should return nothing.

2. ✅ Try pushing again:
   ```bash
   git push origin main --force
   ```

3. ✅ Update your `.env` file with the new API key:
   ```bash
   echo "OPENAI_API_KEY=your-new-key-here" >> python/.env
   ```

## What I Already Fixed

- ✅ Updated `python/openai-test.py` to load API key from environment variable
- ✅ File now uses `python-dotenv` to load from `.env` file
- ✅ No more hardcoded API keys

## Security Best Practices Going Forward

1. **Never commit API keys** - Always use environment variables
2. **Use `.env` files** - Add them to `.gitignore`
3. **Check before committing**:
   ```bash
   git diff --cached | grep -i "sk-"
   ```
4. **Use pre-commit hooks** - Install `detect-secrets` or similar tools

## Need Help?

If you get stuck, you can:
1. Use the backup branch: `git checkout backup-before-fix`
2. Ask for help with the specific error message
3. Consider using GitHub's "Allow secret" option (NOT recommended for real keys)
