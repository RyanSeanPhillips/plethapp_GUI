"""
Rebrand PlethApp to PhysioMetrics

This script updates all remaining references from PlethApp to PhysioMetrics.
Run this from the project root directory.

Usage:
    python _rebrand_to_physiometrics.py
"""

import os
import re
from pathlib import Path

# Files and patterns to update
REPLACEMENTS = {
    # Exact word replacements (case-sensitive)
    r'\bPlethApp\b': 'PhysioMetrics',
    r'\bplethapp\b': 'physiometrics',
    r'\bpleth_app\b': 'physiometrics',

    # Specific patterns
    r'pleth_app\.spec': 'physiometrics.spec',
    r'PlethApp_v': 'PhysioMetrics_v',
    r'plethapp_GUI': 'PhysioMetrics',
}

# File patterns to process
INCLUDE_PATTERNS = [
    '**/*.py',
    '**/*.md',
    '**/*.spec',
    '**/*.bat',
    '**/*.txt',
    '**/*.ui',
    '**/*.cff',
]

# Directories to skip
SKIP_DIRS = {
    '__pycache__',
    '.git',
    'dist',
    'build',
    'venv',
    'env',
    '.venv',
    'node_modules',
}

def should_process_file(file_path: Path) -> bool:
    """Check if file should be processed."""
    # Skip if in excluded directory
    for part in file_path.parts:
        if part in SKIP_DIRS:
            return False

    # Skip this script itself
    if file_path.name == '_rebrand_to_physiometrics.py':
        return False

    # Check if matches include patterns
    for pattern in INCLUDE_PATTERNS:
        if file_path.match(pattern):
            return True

    return False

def process_file(file_path: Path) -> tuple[int, list[str]]:
    """
    Process a single file, applying all replacements.

    Returns:
        (num_replacements, changes_list)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Skip binary files
        return 0, []
    except Exception as e:
        print(f"  [WARN] Could not read {file_path}: {e}")
        return 0, []

    original_content = content
    changes = []

    for pattern, replacement in REPLACEMENTS.items():
        matches = list(re.finditer(pattern, content))
        if matches:
            changes.append(f"    {len(matches)}x {pattern} -> {replacement}")
            content = re.sub(pattern, replacement, content)

    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return len(changes), changes
        except Exception as e:
            print(f"  [WARN] Could not write {file_path}: {e}")
            return 0, []

    return 0, []

def main():
    print("=" * 70)
    print("PhysioMetrics Rebranding Script")
    print("=" * 70)
    print()

    project_root = Path(__file__).parent
    print(f"Project root: {project_root}")
    print()

    # Collect all files to process
    all_files = []
    for pattern in INCLUDE_PATTERNS:
        all_files.extend(project_root.glob(pattern))

    files_to_process = [f for f in all_files if should_process_file(f)]

    print(f"Found {len(files_to_process)} files to process")
    print()

    # Process files
    total_replacements = 0
    modified_files = []

    for file_path in sorted(files_to_process):
        num_changes, changes = process_file(file_path)

        if num_changes > 0:
            rel_path = file_path.relative_to(project_root)
            print(f"[OK] {rel_path}")
            for change in changes:
                print(change)
            modified_files.append(rel_path)
            total_replacements += num_changes

    print()
    print("=" * 70)
    print(f"[OK] Complete! Modified {len(modified_files)} files")
    print(f"  Total replacement groups: {total_replacements}")
    print("=" * 70)
    print()

    # Special file renames
    print("Files that need manual renaming:")
    print("  1. pleth_app.spec -> physiometrics.spec")
    print("  2. Any executables: PlethApp_v*.exe -> PhysioMetrics_v*.exe")
    print()

    # GitHub rename reminder
    print("=" * 70)
    print("GitHub Repository Rename Instructions")
    print("=" * 70)
    print()
    print("To rename your GitHub repository:")
    print("  1. Go to: https://github.com/RyanSeanPhillips/plethapp_GUI")
    print("  2. Click 'Settings' (gear icon)")
    print("  3. Repository name -> Change 'plethapp_GUI' to 'PhysioMetrics'")
    print("  4. Click 'Rename'")
    print()
    print("GitHub automatically redirects old URLs, so:")
    print("  [OK] Old links still work (they redirect)")
    print("  [OK] Local git repos still work")
    print("  [OK] Forks and stars are preserved")
    print()
    print("After renaming, update your local repo:")
    print("  git remote set-url origin https://github.com/RyanSeanPhillips/PhysioMetrics.git")
    print()

if __name__ == '__main__':
    main()
