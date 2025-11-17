@echo off
REM GREMLIN Massive Language Pack Generator (Windows)
REM Generates a 10,000 words/concept pack (~78 MB, 1.86M words)

echo ======================================================================
echo GREMLIN Massive Language Pack Generator
echo ======================================================================
echo.
echo This will generate a ~78 MB language pack with 1,860,000 words.
echo Generation will take 2-5 minutes.
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo Starting generation...
echo.

python generate_language_pack.py --words 10000 --output language_packs/

echo.
echo ======================================================================
echo Done! Your language pack is ready.
echo ======================================================================
echo.
echo To use it with the admin console, run:
echo     python demo/admin_console_tk.py --pack language_packs/language_pack_*.json
echo.
pause
