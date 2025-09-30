@echo off
REM ====================================================================
REM PlethApp Build Script - Creates Windows Executable
REM ====================================================================
REM This script automates the process of building a Windows executable
REM from the PlethApp breath analysis application using PyInstaller.
REM ====================================================================

echo.
echo ====================================================================
echo Building PlethApp Windows Executable
echo ====================================================================
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo ERROR: PyInstaller is not installed!
    echo Please install it with: pip install pyinstaller
    echo.
    pause
    exit /b 1
)

REM Clean previous builds
echo Cleaning previous builds...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"
if exist "__pycache__" rmdir /s /q "__pycache__"

REM Clean Python cache files
for /r %%i in (*.pyc) do del "%%i" 2>nul
for /r %%i in (__pycache__) do rmdir /s /q "%%i" 2>nul

echo.
echo Starting PyInstaller build...
echo This may take several minutes...
echo.

REM Build the executable using the spec file
pyinstaller --clean pleth_app.spec

REM Check if build was successful
if exist "dist\PlethApp.exe" (
    echo.
    echo ====================================================================
    echo BUILD SUCCESSFUL!
    echo ====================================================================
    echo.
    echo Executable created: dist\PlethApp.exe
    echo File size:
    for %%A in ("dist\PlethApp.exe") do echo %%~zA bytes
    echo.
    echo You can now distribute the exe file to users.
    echo The executable is self-contained and doesn't require Python installation.
    echo.
) else (
    echo.
    echo ====================================================================
    echo BUILD FAILED!
    echo ====================================================================
    echo.
    echo Please check the output above for error messages.
    echo Common issues:
    echo - Missing dependencies in requirements.txt
    echo - UI files not found in ui\ directory
    echo - Icon files not found in images\ directory
    echo.
)

echo.
echo Build process completed.
pause