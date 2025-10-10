"""Reduce groupbox fonts from 9pt to 8pt."""
from pathlib import Path

def reduce_fonts(ui_file):
    """Change all 9pt fonts within groupboxes to 8pt."""
    with open(ui_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find groupbox boundaries
    groupboxes = {
        'PeakDetection': None,
        'groupBox': None,
        'groupBox_2': None
    }

    for i, line in enumerate(lines):
        for name in groupboxes.keys():
            if f'<widget class="QGroupBox" name="{name}">' in line:
                # Find closing </widget> for this groupbox
                depth = 1
                start = i
                for j in range(i + 1, len(lines)):
                    if '<widget' in lines[j]:
                        depth += 1
                    elif '</widget>' in lines[j]:
                        depth -= 1
                        if depth == 0:
                            groupboxes[name] = (start, j)
                            break

    print("Found groupboxes:")
    for name, bounds in groupboxes.items():
        if bounds:
            print(f"  {name}: lines {bounds[0]+1} to {bounds[1]+1}")

    # Change 9pt to 8pt within groupboxes, but skip groupbox title (10pt bold)
    changes = 0
    for name, bounds in groupboxes.items():
        if not bounds:
            continue

        start, end = bounds

        # Process lines in this groupbox
        i = start + 1
        while i <= end:
            if '<pointsize>9</pointsize>' in lines[i]:
                # Check if this is part of the groupbox title font (has bold=true nearby)
                # Look at surrounding context
                is_title = False
                for j in range(max(start, i-10), min(end, i+10)):
                    if '<pointsize>10</pointsize>' in lines[j] and '<bold>true</bold>' in lines[j+1:j+5]:
                        is_title = True
                        break

                if not is_title:
                    lines[i] = lines[i].replace('<pointsize>9</pointsize>', '<pointsize>8</pointsize>')
                    changes += 1

            i += 1

    print(f"\nChanged {changes} fonts from 9pt to 8pt")

    # Write back
    with open(ui_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Updated: {ui_file}")

if __name__ == "__main__":
    ui_dir = Path(__file__).parent / "ui"
    ui_file = ui_dir / "pleth_app_layout_02_horizontal.ui"

    reduce_fonts(ui_file)
    print("\nDone! All groupbox widget fonts reduced to 8pt.")
