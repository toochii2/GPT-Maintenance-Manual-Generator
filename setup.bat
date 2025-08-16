@echo off
echo ======================================
echo GPT Maintenance Manual Generator Setup
echo ======================================
echo.

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo Error: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

REM Initialize Git repository if not already initialized
if not exist ".git" (
    echo Initializing Git repository...
    git init
    echo Git repository initialized.
    echo.
) else (
    echo Git repository already exists.
    echo.
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org/
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    echo Virtual environment created.
    echo.
) else (
    echo Virtual environment already exists.
    echo.
)

REM Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ======================================
echo Setup Instructions
echo ======================================
echo.
echo 1. Create a new repository on GitHub:
echo    - Go to https://github.com/new
echo    - Repository name: gpt-maintenance-manual-generator
echo    - Description: AI-powered tool that converts maintenance videos into structured manuals
echo    - Make it Public
echo    - Don't initialize with README (we already have one)
echo.
echo 2. After creating the repository, run these commands:
echo    git add .
echo    git commit -m "Initial commit: GPT Maintenance Manual Generator"
echo    git branch -M main
echo    git remote add origin https://github.com/Toochii2/gpt-maintenance-manual-generator.git
echo    git push -u origin main
echo.
echo 3. Enable GitHub Pages:
echo    - Go to your repository Settings
echo    - Scroll to Pages section
echo    - Source: Deploy from a branch
echo    - Branch: main
echo    - Folder: /docs
echo    - Save
echo.
echo 4. Your website will be available at:
echo    https://Toochii2.github.io/gpt-maintenance-manual-generator
echo.
echo 5. Update the following in your files:
echo    - Replace "yourusername" with your GitHub username in:
echo      * README.md
echo      * docs/index.html
echo      * Any other references
echo.
echo ======================================
echo Quick Commands Reference
echo ======================================
echo.
echo To run the application:
echo    venv\Scripts\activate
echo    streamlit run app.py
echo.
echo To update dependencies:
echo    pip install -r requirements.txt
echo.
echo To push changes:
echo    git add .
echo    git commit -m "Your commit message"
echo    git push
echo.
echo ======================================
echo Setup Complete!
echo ======================================
echo.
pause
