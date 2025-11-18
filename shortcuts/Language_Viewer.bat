@echo off
REM GREMLIN Language Pack Viewer Launcher
REM Place this in F:\dev\Admin_Console_Shortcuts\

echo ===============================================
echo    GREMLIN Language Pack Viewer
echo ===============================================
echo.

REM Navigate to GREMLIN directory
cd /d F:\dev\GREMLIN_Claude_Code_Web_track

if not exist "language_pack_viewer.py" (
    echo ERROR: Cannot find GREMLIN installation
    echo Expected: F:\dev\GREMLIN_Claude_Code_Web_track
    echo.
    pause
    exit /b 1
)

echo Starting language pack viewer...
echo.

py -3.12 language_pack_viewer.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch viewer
    echo.
    pause
)
