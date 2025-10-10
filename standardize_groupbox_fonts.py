"""Standardize fonts within groupboxes to 9pt non-bold."""
import re
from pathlib import Path

def standardize_fonts(ui_file):
    """
    Standardize all fonts within the three groupboxes to 9pt non-bold.
    Groupboxes: PeakDetection, groupBox (Filtering), groupBox_2 (Channel Selection)
    """
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

    # Pattern to match font definitions
    font_pattern = re.compile(r'(<property name="font">.*?</property>)', re.DOTALL)

    # Standard font to use (9pt, non-bold, no family specified to use default)
    standard_font = """<property name="font">
             <font>
              <pointsize>9</pointsize>
              <bold>false</bold>
             </font>
            </property>"""

    # Process each groupbox
    changes_made = 0
    for name, bounds in groupboxes.items():
        if not bounds:
            continue

        start, end = bounds

        # Process lines in this groupbox (skip the groupbox title itself)
        i = start + 1
        while i <= end:
            line = lines[i]

            # Check if this is a font property
            if '<property name="font">' in line:
                # Find the end of this font property
                j = i
                while '</property>' not in lines[j]:
                    j += 1

                # Extract current font block
                font_block = ''.join(lines[i:j+1])

                # Check if this is the groupbox title font (has bold=true and pointsize=10)
                if '<bold>true</bold>' in font_block and '<pointsize>10</pointsize>' in font_block:
                    # Skip groupbox title font
                    i = j + 1
                    continue

                # Get indentation from first line
                indent = len(line) - len(line.lstrip())
                indented_font = '\n'.join(' ' * indent + l if l.strip() else ''
                                          for l in standard_font.split('\n'))

                # Replace the font block
                lines[i:j+1] = [indented_font + '\n']
                changes_made += 1

                # Adjust end pointer
                end = end - (j - i) + 1
                i += 1
            else:
                i += 1

    print(f"\nMade {changes_made} font changes")

    # Write back
    with open(ui_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Updated: {ui_file}")

if __name__ == "__main__":
    ui_dir = Path(__file__).parent / "ui"
    ui_file = ui_dir / "pleth_app_layout_02_horizontal.ui"

    standardize_fonts(ui_file)
    print("\nDone! All fonts in groupboxes standardized to 9pt non-bold.")
