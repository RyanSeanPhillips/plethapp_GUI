@echo off
REM Launch PlethApp in testing mode WITH LINE PROFILING
REM Auto-loads: C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\25121004.abf
REM Auto-sets: Analyze Channel = 0, Stim Channel = 7
REM
REM IMPORTANT: This will be 10-100x SLOWER than normal due to profiling overhead!
REM            Only use this when you want to measure performance.

echo ============================================================
echo  PlethApp PROFILING MODE
echo ============================================================
echo.
echo WARNING: App will run 10-100x slower due to profiling!
echo.
echo Instructions:
echo   1. App will load test file automatically
echo   2. Trigger the operation you want to profile (e.g., export)
echo   3. Close the app - results will be saved automatically
echo.
echo Starting profiler...
echo ============================================================
echo.

REM Create profiling_results directory if it doesn't exist
if not exist profiling_results mkdir profiling_results

REM Run the profiler
cmd /c "set PLETHAPP_TESTING=1&& kernprof -l run_debug.py"

echo.
echo ============================================================
echo  Profiling Complete! Generating report...
echo ============================================================
echo.

REM Generate timestamp for filename (YYYYMMDD_HHMMSS format)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%

REM Set output filename
set output_file=profiling_results\profile_%timestamp%.txt

REM Generate and save profiling report
python -m line_profiler run_debug.py.lprof > "%output_file%"

REM Also display in terminal
echo Results saved to: %output_file%
echo.
echo ============================================================
echo  Top Lines by Time (showing first 100 lines):
echo ============================================================
echo.
type "%output_file%" | more

echo.
echo ============================================================
echo  Full results saved to: %output_file%
echo ============================================================
echo.
pause
