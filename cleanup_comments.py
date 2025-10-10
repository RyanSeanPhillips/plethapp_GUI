#!/usr/bin/env python3
"""
Remove large blocks of commented-out code from Python files.
Keeps: single-line comments, docstrings, important markers (TODO, FIXME, NOTE, etc.)
Removes: Multi-line blocks of commented code (>5 consecutive lines)
"""

import re
from pathlib import Path


def is_important_comment(line):
    """Check if a comment line should be preserved."""
    comment_text = line.strip().lstrip('#').strip().upper()

    # Preserve important markers
    important_markers = [
        'TODO', 'FIXME', 'NOTE', 'WARNING', 'IMPORTANT', 'XXX', 'HACK',
        'BUG', 'ISSUE', 'DEPRECATED', 'TESTING MODE'
    ]

    if any(marker in comment_text for marker in important_markers):
        return True

    # Preserve section headers (comments with many ==== or ----)
    if '====' in line or '----' in line:
        return True

    # Preserve short explanatory comments (no code-like patterns)
    if not any(pattern in line for pattern in ['def ', 'class ', 'return ', 'self.', 'import ', '= ', '== ', '!=']):
        return True

    return False


def clean_commented_blocks(file_path, min_block_size=5, dry_run=False):
    """Remove blocks of consecutive commented lines that look like old code."""

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    in_block = False
    block_start = 0
    block_lines = []
    removed_blocks = []

    for i, line in enumerate(lines):
        # Check if this is a commented line (with proper indentation)
        if re.match(r'^(\s*)#', line):
            # Check if it's an important comment to preserve
            if is_important_comment(line):
                # Preserve this line even if in a block
                if in_block and len(block_lines) >= min_block_size:
                    # End current block, remove it
                    removed_blocks.append((block_start, block_start + len(block_lines) - 1, len(block_lines)))
                    in_block = False
                    block_lines = []
                cleaned_lines.append(line)
            else:
                # Start or continue a block
                if not in_block:
                    in_block = True
                    block_start = i
                    block_lines = [line]
                else:
                    block_lines.append(line)
        else:
            # Not a comment line
            if in_block:
                # End the block
                if len(block_lines) >= min_block_size:
                    # This was a large block - remove it
                    removed_blocks.append((block_start, block_start + len(block_lines) - 1, len(block_lines)))
                else:
                    # Small block - keep it (might be useful comments)
                    cleaned_lines.extend(block_lines)

                in_block = False
                block_lines = []

            # Always keep non-comment lines
            cleaned_lines.append(line)

    # Handle block at end of file
    if in_block and len(block_lines) >= min_block_size:
        removed_blocks.append((block_start, block_start + len(block_lines) - 1, len(block_lines)))
    elif in_block:
        cleaned_lines.extend(block_lines)

    # Report
    print(f"\n{file_path}:")
    print(f"  Original: {len(lines)} lines")
    print(f"  Cleaned: {len(cleaned_lines)} lines")
    print(f"  Removed: {len(lines) - len(cleaned_lines)} lines ({len(removed_blocks)} blocks)")

    if removed_blocks:
        print(f"  Removed blocks:")
        for start, end, size in removed_blocks[:10]:  # Show first 10
            print(f"    Lines {start+1}-{end+1} ({size} lines)")
        if len(removed_blocks) > 10:
            print(f"    ... and {len(removed_blocks) - 10} more blocks")

    # Write cleaned file
    if not dry_run:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        print(f"  [OK] Cleaned!")
    else:
        print(f"  (DRY RUN - no changes made)")

    return len(lines) - len(cleaned_lines)


def main():
    print("=" * 60)
    print("Cleaning up commented-out code blocks")
    print("=" * 60)

    # Files to clean
    files_to_clean = [
        Path("main.py"),
        Path("core/filters.py"),
        Path("core/metrics.py"),
        Path("core/peaks.py"),
        Path("core/plotting.py"),
    ]

    total_removed = 0

    # First, do a dry run to show what will be removed
    print("\nDRY RUN - Preview of changes:")
    print("-" * 60)
    for file_path in files_to_clean:
        if file_path.exists():
            clean_commented_blocks(file_path, min_block_size=5, dry_run=True)
        else:
            print(f"\n{file_path}: File not found")

    # Ask for confirmation
    print("\n" + "=" * 60)
    response = input("Proceed with cleanup? (yes/no): ").strip().lower()

    if response == 'yes':
        print("\nCleaning files...")
        print("-" * 60)
        for file_path in files_to_clean:
            if file_path.exists():
                removed = clean_commented_blocks(file_path, min_block_size=5, dry_run=False)
                total_removed += removed

        print("\n" + "=" * 60)
        print(f"DONE! Removed {total_removed} lines total")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Test the application: python run_testing.bat")
        print("  2. If everything works, commit the changes")
        print("  3. If something broke, revert: git checkout main.py core/")
    else:
        print("\nCleanup cancelled.")


if __name__ == '__main__':
    main()
