@echo off
REM Launch PlethApp in testing mode
REM Auto-loads: C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\25121004.abf
REM Auto-sets: Analyze Channel = 0, Stim Channel = 7

echo Starting PlethApp in TESTING MODE...
cmd /c "set PLETHAPP_TESTING=1&& python run_debug.py"
