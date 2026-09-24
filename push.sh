#!/bin/bash
set -e

REPO_URL="git@github.com:niweshsah/Amzon-ML-Challenge-2026.git"

# 1. Initialize Git if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository in $(pwd)..."
    git init
    git branch -M main
    git remote add origin "$REPO_URL"
fi

# 2. Ensure remote uses the SSH URL
git remote set-url origin "$REPO_URL"

# 3. Create README.md if it does not exist
if [ ! -f "README.md" ]; then
    echo "# Amzon-ML-Challenge-2026" > README.md
fi

# 4. Exclude push.sh itself from being pushed to GitHub
if ! grep -q "^push.sh$" .gitignore 2>/dev/null; then
    echo "push.sh" >> .gitignore
fi

# 5. Stage files, commit, and push
git add .
git diff-index --quiet HEAD || git commit -m "Update code and scripts"

echo "Pushing workspace to GitHub using forwarded SSH key..."
git push -u origin main

echo "Push successful!"
