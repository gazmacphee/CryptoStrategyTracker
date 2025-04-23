@echo off
echo ======================================================
echo CryptoApp Windows Setup Helper
echo ======================================================
echo.
echo This script will install the required npm package for data downloading.
echo.

REM Check if node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Node.js is not installed or not in PATH. Please install Node.js from https://nodejs.org/
    echo After installing, try running this script again.
    pause
    exit /b 1
)

echo Node.js is installed. Version:
node --version
echo.

REM Find npm by looking in common locations
set NPM_FOUND=0
set NPM_PATH=

REM Try to find npm in PATH first
where npm >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set NPM_FOUND=1
    set NPM_PATH=npm
    echo Found npm in PATH
    goto :NPM_CHECK_DONE
)

REM Try to find npm in common Node.js installation locations
for %%P in (
    "%ProgramFiles%\nodejs\npm.cmd"
    "%ProgramFiles(x86)%\nodejs\npm.cmd"
    "%APPDATA%\npm\npm.cmd"
    "%LOCALAPPDATA%\Programs\nodejs\npm.cmd"
) do (
    if exist %%P (
        set NPM_FOUND=1
        set NPM_PATH=%%P
        echo Found npm at: %%P
        goto :NPM_CHECK_DONE
    )
)

:NPM_CHECK_DONE
if %NPM_FOUND% EQU 0 (
    echo npm not found in common locations.
    echo Let's try using npx instead...
    
    where npx >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo npx found, we'll use it instead.
        set NPM_FOUND=1
        set NPM_PATH=npx
    ) else (
        echo npx not found either.
        echo The application will automatically use direct downloads instead.
        echo.
        goto :SKIP_INSTALL
    )
)

REM Show npm version if found
if %NPM_FOUND% EQU 1 (
    if "%NPM_PATH%"=="npm" (
        echo npm is installed. Version:
        npm --version
    ) else if "%NPM_PATH%"=="npx" (
        echo npx is installed. Version: 
        npx --version
    ) else (
        echo npm is installed at %NPM_PATH%
    )
    echo.
)

REM Create package.json if it doesn't exist
if not exist package.json (
    echo Creating package.json file...
    echo {"name": "crypto-app", "version": "1.0.0", "private": true} > package.json
    echo Created package.json file.
    echo.
)

REM Install using the found npm or npx path
echo Installing binance-historical-data package...
if "%NPM_PATH%"=="npm" (
    call npm install binance-historical-data --no-save
) else if "%NPM_PATH%"=="npx" (
    call npx binance-historical-data --version
) else (
    call %NPM_PATH% install binance-historical-data --no-save
)

if %ERRORLEVEL% NEQ 0 (
    echo Failed to install binance-historical-data package.
    echo Don't worry - the application will automatically use direct downloads instead.
    echo.
) else (
    echo Successfully installed binance-historical-data package.
    echo.
)

:SKIP_INSTALL
REM Create an npm.bat file for the Python script to use
echo @echo off > npm.bat
echo REM This is a helper script created by setup_windows.bat >> npm.bat
if %NPM_FOUND% EQU 1 (
    if "%NPM_PATH%"=="npm" (
        echo npm %%* >> npm.bat
    ) else if "%NPM_PATH%"=="npx" (
        echo npx %%* >> npm.bat
    ) else (
        echo call %NPM_PATH% %%* >> npm.bat
    )
) else (
    echo echo npm not available. The application will use direct downloads. >> npm.bat
    echo exit /b 1 >> npm.bat
)

echo.
echo Created helper script npm.bat that the application will use.
echo.
echo Setup completed. You can now run the application.
echo Even if npm installation failed, the application will use direct downloads.
echo.
echo Press any key to exit...
pause > nul