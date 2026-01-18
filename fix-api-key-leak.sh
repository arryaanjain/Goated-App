#!/bin/bash

# Fix API Key Leak in Git History
# This script removes the hardcoded API key from python/openai-test.py

echo "🔒 Fixing API key leak in Git history..."
echo ""
echo "⚠️  WARNING: This will rewrite Git history!"
echo "   Make sure you have a backup of your work."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1: Creating a backup branch..."
git branch backup-before-fix

echo ""
echo "Step 2: Removing API key from history using git filter-branch..."

# Use git filter-branch to rewrite the file in history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch python/openai-test.py' \
  --prune-empty --tag-name-filter cat -- --all

echo ""
echo "Step 3: Cleaning up..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Step 4: Re-adding the fixed file..."
git add python/openai-test.py
git commit -m "Fix: Remove hardcoded API key from openai-test.py"

echo ""
echo "✅ Done! The API key has been removed from Git history."
echo ""
echo "Next steps:"
echo "1. Revoke the exposed API key at: https://platform.openai.com/api-keys"
echo "2. Generate a new API key"
echo "3. Add it to python/.env file"
echo "4. Force push to GitHub: git push origin main --force"
echo ""
echo "⚠️  Note: Force push will rewrite history on GitHub!"
echo "   Make sure no one else is working on this branch."
