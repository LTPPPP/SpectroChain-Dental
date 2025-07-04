@echo off
REM SpectroChain-Dental Benchmark Runner Script for Windows
REM Chạy benchmark với các options khác nhau

echo 🔬 SpectroChain-Dental Benchmark Suite
echo ======================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo 📥 Installing requirements...
pip install -r requirements.txt

REM Parse command line arguments
set MODE=%1
if "%MODE%"=="" set MODE=full

if "%MODE%"=="performance" (
    echo 🚀 Running performance benchmark...
    python main.py --mode performance
) else if "%MODE%"=="security" (
    echo 🚀 Running security benchmark...
    python main.py --mode security
) else if "%MODE%"=="accuracy" (
    echo 🚀 Running accuracy benchmark...
    python main.py --mode accuracy
) else if "%MODE%"=="full" (
    echo 🚀 Running full benchmark suite...
    python main.py --mode full
) else if "%MODE%"=="demo" (
    echo 🎭 Running demo...
    python demo.py
) else if "%MODE%"=="help" (
    echo Usage: %0 [performance^|security^|accuracy^|full^|demo^|help]
    echo.
    echo Options:
    echo   performance  - Run only performance tests
    echo   security     - Run only security analysis
    echo   accuracy     - Run only accuracy evaluation
    echo   full         - Run complete benchmark suite ^(default^)
    echo   demo         - Run demonstration script
    echo   help         - Show this help message
) else (
    echo ❓ Unknown option: %MODE%
    echo Use '%0 help' for usage information
    pause
    exit /b 1
)

if errorlevel 1 (
    echo ❌ Benchmark failed
    pause
    exit /b 1
) else (
    echo ✅ Benchmark completed successfully
)

echo.
echo 📊 Benchmark completed! Check the results\ directory for reports.
echo 🌐 Open the HTML report in your browser for detailed analysis.

pause
