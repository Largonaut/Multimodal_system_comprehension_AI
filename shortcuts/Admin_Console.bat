@echo off
REM GREMLIN Admin Console Launcher
REM Place this in F:\dev\Admin_Console_Shortcuts\

echo ===============================================
echo    GREMLIN Admin Console
echo ===============================================
echo.

REM Navigate to GREMLIN directory
cd /d F:\dev\GREMLIN_Claude_Code_Web_track

if not exist "demo\admin_console_tk.py" (
    echo ERROR: Cannot find GREMLIN installation
    echo Expected: F:\dev\GREMLIN_Claude_Code_Web_track
    echo.
    pause
    exit /b 1
)

echo Starting admin console...
echo.

py -3.12 demo\admin_console_tk.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch admin console
    echo.
    pause
)
