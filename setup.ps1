# Maintenance Manual Generator Setup Script
# PowerShell version for modern Windows systems

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "GPT Maintenance Manual Generator Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
try {
    $gitVersion = git --version 2>$null
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Error: Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Initialize Git repository if not already initialized
if (-not (Test-Path ".git")) {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✓ Git repository initialized." -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "✓ Git repository already exists." -ForegroundColor Green
    Write-Host ""
}

# Check if Python is installed
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from https://python.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✓ Virtual environment created." -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "✓ Virtual environment already exists." -ForegroundColor Green
    Write-Host ""
}

# Activate virtual environment and install dependencies
Write-Host "Activating virtual environment and installing dependencies..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Setup Instructions" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Create a new repository on GitHub:" -ForegroundColor White
Write-Host "   - Go to https://github.com/new" -ForegroundColor Gray
Write-Host "   - Repository name: gpt-maintenance-manual-generator" -ForegroundColor Gray
Write-Host "   - Description: AI-powered tool that converts maintenance videos into structured manuals" -ForegroundColor Gray
Write-Host "   - Make it Public" -ForegroundColor Gray
Write-Host "   - Don't initialize with README (we already have one)" -ForegroundColor Gray
Write-Host ""

Write-Host "2. After creating the repository, run these commands:" -ForegroundColor White
Write-Host "   git add ." -ForegroundColor Green
Write-Host "   git commit -m 'Initial commit: GPT Maintenance Manual Generator'" -ForegroundColor Green
Write-Host "   git branch -M main" -ForegroundColor Green
Write-Host "   git remote add origin https://github.com/Toochii2/gpt-maintenance-manual-generator.git" -ForegroundColor Green
Write-Host "   git push -u origin main" -ForegroundColor Green
Write-Host ""

Write-Host "3. Enable GitHub Pages:" -ForegroundColor White
Write-Host "   - Go to your repository Settings" -ForegroundColor Gray
Write-Host "   - Scroll to Pages section" -ForegroundColor Gray
Write-Host "   - Source: Deploy from a branch" -ForegroundColor Gray
Write-Host "   - Branch: main" -ForegroundColor Gray
Write-Host "   - Folder: /docs" -ForegroundColor Gray
Write-Host "   - Save" -ForegroundColor Gray
Write-Host ""

Write-Host "4. Your website will be available at:" -ForegroundColor White
Write-Host "   https://Toochii2.github.io/gpt-maintenance-manual-generator" -ForegroundColor Blue
Write-Host ""

Write-Host "5. Update the following in your files:" -ForegroundColor White
Write-Host "   - Replace 'yourusername' with your GitHub username in:" -ForegroundColor Gray
Write-Host "     * README.md" -ForegroundColor Gray
Write-Host "     * docs/index.html" -ForegroundColor Gray
Write-Host "     * Any other references" -ForegroundColor Gray
Write-Host ""

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Quick Commands Reference" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "To run the application:" -ForegroundColor White
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "   streamlit run app.py" -ForegroundColor Green
Write-Host ""

Write-Host "To update dependencies:" -ForegroundColor White
Write-Host "   pip install -r requirements.txt" -ForegroundColor Green
Write-Host ""

Write-Host "To push changes:" -ForegroundColor White
Write-Host "   git add ." -ForegroundColor Green
Write-Host "   git commit -m 'Your commit message'" -ForegroundColor Green
Write-Host "   git push" -ForegroundColor Green
Write-Host ""

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to continue"
