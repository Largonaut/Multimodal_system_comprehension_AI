@echo off
REM Update GREMLIN Shortcuts
REM Run this after git pull to update shortcuts in Admin_Console_Shortcuts

echo ============================================================
echo   GREMLIN Shortcut Updater
echo ============================================================
echo.

REM Check if shortcuts directory exists
if not exist "shortcuts\" (
    echo ERROR: shortcuts directory not found
    echo Run this from the GREMLIN root directory
    pause
    exit /b 1
)

REM Check if destination exists
set DEST=F:\dev\Admin_Console_Shortcuts
if not exist "%DEST%\" (
    echo Creating destination directory: %DEST%
    mkdir "%DEST%"
)

echo Copying updated shortcuts to %DEST%
echo.

copy /Y shortcuts\Admin_Console.bat "%DEST%\"
copy /Y shortcuts\Language_Generator.bat "%DEST%\"
copy /Y shortcuts\Language_Viewer.bat "%DEST%\"

echo.
echo ✓ Shortcuts updated successfully!
echo.
pause
