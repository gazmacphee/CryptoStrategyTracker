@echo off
set PYTHONIOENCODING=utf-8
echo ========================================================================
echo Starting Crypto Trading App with Enhanced Setup...
echo ========================================================================

:: Activate virtual environment if it exists
if exist .\venv\Scripts\activate (
  echo Activating virtual environment...
  call .\venv\Scripts\activate
) else (
  echo No virtual environment found. Using system Python.
)

:: Create required database tables first
echo [1/5] Ensuring all database tables are created...
python ensure_tables.py
if %ERRORLEVEL% NEQ 0 (
  echo ERROR: Failed to create database tables!
  echo Trying fallback method...
  python database.py
)

:: Create economic tables specially to fix Windows encoding issues
echo [2/5] Ensuring economic indicator tables exist...
python create_economic_tables.py

:: Fix for Windows encoding issues
echo [3/5] Setting up Windows-specific configurations...
set PYTHONIOENCODING=utf-8
echo Environment variable PYTHONIOENCODING=utf-8 set to fix encoding issues

:: Start the Process Manager
echo [4/5] Starting Process Manager...
start "Process Manager" cmd /k python process_manager.py run

:: Give process manager time to initialize
echo Waiting for Process Manager to initialize...
timeout /t 5 /nobreak > nul

:: Start the main application
echo [5/5] Starting Streamlit Application...
start "Crypto Trading App" cmd /k python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo ========================================================================
echo All components started successfully!
echo ========================================================================
echo Application URLs:
echo - Main App: http://localhost:8501
echo ========================================================================
echo.
echo Press any key to exit this window (app will continue running)...
pause > nul