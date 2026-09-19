@echo off
setlocal
cd /d "%~dp0"
echo Retrying papers 04-25 with MDPI static PDF fallback...
echo.
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 download_papers.py --start 4 --limit 22
) else (
    python download_papers.py --start 4 --limit 22
)
echo.
echo Finished.
pause
