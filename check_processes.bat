@echo off
echo ================================================================================
echo CryptoStrategyTracker Process Status Check
echo ================================================================================
echo.

echo RUNNING PYTHON PROCESSES:
echo ------------------------
wmic process where "name='python.exe'" get processid,commandline

echo.
echo PROCESS MANAGER STATUS:
echo ---------------------
python process_manager.py status

echo.
echo LATEST LOG ENTRIES:
echo -----------------
echo Process Manager Log (last 5 lines):
type process_manager.log | findstr /n ".*" | findstr /r "^[0-9]*[0-9][0-9][0-9][0-9][0-9]:" | tail -5

echo.
echo ================================================================================
echo COMMAND REFERENCE:
echo -----------------
echo Start all processes:    python process_manager.py start
echo Start and keep running: python process_manager.py run
echo Stop all processes:     python process_manager.py stop
echo Restart all processes:  python process_manager.py restart
echo Monitor all processes:  python process_manager.py monitor
echo.
echo To check status again:  check_processes.bat
echo ================================================================================

pause