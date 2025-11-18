@echo off
REM GREMLIN Language Pack Generator Launcher
REM Place this in F:\dev\Admin_Console_Shortcuts\

echo ===============================================
echo    GREMLIN Language Pack Generator
echo ===============================================
echo.

REM Navigate to GREMLIN directory
cd /d F:\dev\GREMLIN_Claude_Code_Web_track

if not exist "language_pack_generator_gui.py" (
    echo ERROR: Cannot find GREMLIN installation
    echo Expected: F:\dev\GREMLIN_Claude_Code_Web_track
    echo.
    pause
    exit /b 1
)

echo Starting language pack generator...
echo.

python language_pack_generator_gui.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch generator
    echo.
    pause
)
