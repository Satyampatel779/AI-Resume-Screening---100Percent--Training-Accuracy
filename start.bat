@echo off
echo.
echo ========================================
echo   AI Resume Screening System
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo Python is installed. Starting setup...
echo.

REM Run the setup and training script
python setup_and_train.py

if %errorlevel% neq 0 (
    echo.
    echo Setup failed! Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.
set /p start_app="Do you want to start the web application now? (y/n): "

if /i "%start_app%"=="y" (
    echo.
    echo Starting Flask web application...
    echo Open your browser and go to: http://localhost:5000
    echo Press Ctrl+C to stop the server
    echo.
    python app.py
) else (
    echo.
    echo To start the application later, run: python app.py
    echo.
)

pause
