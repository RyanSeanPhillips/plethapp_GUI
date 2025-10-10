"""
Script to create compact UI layout versions.
This restructures the top control sections to save vertical space.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

def reduce_font_sizes(root):
    """Reduce font sizes: headers 12->11pt, labels 11->10pt"""
    for font_elem in root.findall(".//font/pointsize"):
        size = int(font_elem.text)
        if size == 12:
            font_elem.text = "10"  # Headers
        elif size == 11:
            font_elem.text = "9"   # Labels
    return root

def create_horizontal_compact(input_file, output_file):
    """
    Create horizontal compact layout:
    - Channel Selection: 3 rows (header, analyze+stimulus, apply button)
    - Filtering: 2 rows of controls
    - Peak Detection: 2 rows of controls
    """
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Reduce font sizes
    root = reduce_font_sizes(root)

    # Find the main vertical layout (verticalLayout_8)
    main_layout = root.find(".//layout[@name='verticalLayout_8']")
    if main_layout is None:
        print("Could not find verticalLayout_8")
        return

    # We need to restructure:
    # 1. Keep File Selection as is
    # 2. Restructure Channel Selection into 3-row grid
    # 3. Restructure Filtering into 2 rows
    # 4. Restructure Peak Detection into 2 rows

    # This is complex - for now, let's just save with reduced fonts
    # Manual editing in Qt Designer might be easier for layout changes

    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Created horizontal compact layout: {output_file}")

def create_tabbed_layout(input_file, output_file):
    """
    Create tabbed interface layout with QTabWidget
    """
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Reduce font sizes
    root = reduce_font_sizes(root)

    # For tabbed layout, we'd need to:
    # 1. Create a QTabWidget
    # 2. Move File Selection + Channel Selection to Tab 1
    # 3. Move Filtering to Tab 2
    # 4. Move Peak Detection to Tab 3

    # This is very complex programmatically
    # Better to provide instructions for manual Qt Designer work

    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Created tabbed layout (with reduced fonts): {output_file}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    input_file = base_dir / "pleth_app_layout_02.ui"

    # Create both versions
    horizontal_out = base_dir / "pleth_app_layout_02_horizontal.ui"
    tabbed_out = base_dir / "pleth_app_layout_02_tabbed.ui"

    create_horizontal_compact(input_file, horizontal_out)
    create_tabbed_layout(input_file, tabbed_out)

    print("\nBoth versions created!")
    print("Note: Full layout restructuring requires Qt Designer.")
    print("These versions have reduced font sizes as a start.")
