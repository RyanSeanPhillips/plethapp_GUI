#!/usr/bin/env python3
"""
Debug launcher for PlethApp
Use this to test the application before building the executable
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Setup the environment for running the application."""
    # Add the current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

    # Change to the application directory
    os.chdir(current_dir)

def check_imports():
    """Check if all required modules can be imported."""
    print("Checking imports...")

    imports = [
        ('PyQt6', 'PyQt6.QtWidgets'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('core.state', 'core.state'),
        ('core.abf_io', 'core.abf_io'),
        ('core.filters', 'core.filters'),
        ('core.plotting', 'core.plotting'),
        ('core.peaks', 'core.peaks'),
        ('core.metrics', 'core.metrics'),
    ]

    failed_imports = []

    for name, module in imports:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed_imports.append(name)

    if failed_imports:
        print(f"\nWARNING: Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("All imports successful!")
        return True

def main():
    """Main debug launcher."""
    print("="*50)
    print("PlethApp Debug Launcher")
    print("="*50)

    setup_environment()

    if not check_imports():
        print("\nSome imports failed. The application may not work correctly.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    print("\nLaunching PlethApp...")
    print("Close the application window to return to this prompt.")
    print("-" * 50)

    try:
        # Import and run the main application
        from main import main as app_main
        app_main()

    except Exception as e:
        print(f"\nERROR: Application crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nApplication closed normally.")

if __name__ == '__main__':
    main()