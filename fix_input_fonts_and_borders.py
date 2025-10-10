"""Fix QLineEdit and QComboBox fonts, and make groupbox borders white."""
import re
from pathlib import Path

def fix_fonts_and_borders(ui_file):
    """
    1. Standardize all QLineEdit and QComboBox fonts within groupboxes to 9pt non-bold
    2. Add white border styling to all groupboxes
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

    # Standard font for input widgets
    standard_font = """<property name="font">
             <font>
              <pointsize>9</pointsize>
              <bold>false</bold>
             </font>
            </property>"""

    # Process QLineEdit and QComboBox widgets in each groupbox
    changes_made = 0
    for name, bounds in groupboxes.items():
        if not bounds:
            continue

        start, end = bounds
        i = start + 1

        while i <= end:
            line = lines[i]

            # Check if this is a QLineEdit or QComboBox widget
            if ('<widget class="QLineEdit"' in line or
                '<widget class="QComboBox"' in line):

                # Find the closing tag for this widget
                widget_start = i
                depth = 1
                j = i + 1
                while j < len(lines) and depth > 0:
                    if '<widget' in lines[j]:
                        depth += 1
                    elif '</widget>' in lines[j]:
                        depth -= 1
                    j += 1
                widget_end = j - 1

                # Check if widget already has a font property
                has_font = False
                font_start = None
                font_end = None

                for k in range(widget_start, widget_end + 1):
                    if '<property name="font">' in lines[k]:
                        has_font = True
                        font_start = k
                        # Find end of font property
                        while '</property>' not in lines[k]:
                            k += 1
                        font_end = k
                        break

                # Get indentation for the widget's properties
                indent = 0
                for k in range(widget_start + 1, widget_end):
                    if '<property' in lines[k]:
                        indent = len(lines[k]) - len(lines[k].lstrip())
                        break

                if has_font:
                    # Replace existing font
                    indented_font = '\n'.join(' ' * indent + l if l.strip() else ''
                                              for l in standard_font.split('\n'))
                    lines[font_start:font_end+1] = [indented_font + '\n']
                    end = end - (font_end - font_start)
                    changes_made += 1
                else:
                    # Add font property after widget opening tag
                    indented_font = '\n'.join(' ' * indent + l if l.strip() else ''
                                              for l in standard_font.split('\n'))
                    lines.insert(widget_start + 1, indented_font + '\n')
                    end += 1
                    changes_made += 1

                i = widget_start + 1
            else:
                i += 1

    print(f"\nMade {changes_made} font changes in QLineEdit and QComboBox widgets")

    # Now add white borders to groupboxes
    groupbox_style = """<property name="styleSheet">
         <string notr="true">QGroupBox {
    border: 1px solid #ffffff;
    border-radius: 4px;
    margin-top: 0.5em;
    padding-top: 0.5em;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px 0 3px;
}</string>
        </property>"""

    border_changes = 0
    for name, bounds in groupboxes.items():
        if not bounds:
            continue

        start, _ = bounds

        # Check if groupbox already has styleSheet
        has_stylesheet = False
        for i in range(start, start + 20):  # Check first 20 lines of groupbox
            if '<property name="styleSheet">' in lines[i]:
                has_stylesheet = True
                break

        if not has_stylesheet:
            # Find where to insert (after title property)
            insert_pos = None
            for i in range(start, start + 30):
                if '<property name="title">' in lines[i]:
                    # Find end of title property
                    while '</property>' not in lines[i]:
                        i += 1
                    insert_pos = i + 1
                    break

            if insert_pos:
                # Get indentation from title property line
                indent = 8  # Standard groupbox property indentation
                indented_style = '\n'.join(' ' * indent + l if l.strip() else ''
                                           for l in groupbox_style.split('\n'))
                lines.insert(insert_pos, indented_style + '\n')
                border_changes += 1

    print(f"Added white borders to {border_changes} groupboxes")

    # Write back
    with open(ui_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"\nUpdated: {ui_file}")

if __name__ == "__main__":
    ui_dir = Path(__file__).parent / "ui"
    ui_file = ui_dir / "pleth_app_layout_02_horizontal.ui"

    fix_fonts_and_borders(ui_file)
    print("\nDone! Fixed input widget fonts and added white groupbox borders.")
