@echo off
echo ================================================================================
echo CryptoStrategyTracker - Windows Startup
echo ================================================================================
echo.

REM Ensure we're using long path names to avoid path truncation issues
setlocal enableextensions

REM First make sure any existing processes are stopped
echo Stopping any existing processes...
python process_manager.py stop --force

REM Start the application
echo.
echo Starting CryptoStrategyTracker...
python reset_and_start.py

echo.
echo =================================================================================
echo CryptoStrategyTracker is now running!
echo =================================================================================
echo Access the web interface at: http://localhost:5000
echo Check process status with: check_processes.bat
echo Stop all processes with:   python process_manager.py stop
echo =================================================================================

REM Keep the window open
pause