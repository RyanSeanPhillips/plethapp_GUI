"""Restore Analysis tab to absolute positioning (remove layout)."""
from pathlib import Path
import xml.etree.ElementTree as ET

def restore_analysis_tab(ui_file):
    """
    Remove the VBoxLayout and restore absolute geometry positioning.
    """
    # Parse the UI file
    tree = ET.parse(ui_file)
    root = tree.getroot()

    # Find the Analysis tab widget
    analysis_tab = None
    for widget in root.iter('widget'):
        if widget.get('name') == 'DataCurration':  # Analysis tab
            analysis_tab = widget
            break

    if analysis_tab is None:
        print("ERROR: Could not find Analysis tab (DataCurration)")
        return

    print("Found Analysis tab")

    # Find the layout element
    layout = None
    for child in list(analysis_tab):
        if child.tag == 'layout' and child.get('name') == 'analysisTabLayout':
            layout = child
            break

    if layout is None:
        print("No layout found - file may already be in absolute positioning mode")
        return

    # Extract all widgets from the layout
    widgets = []

    def extract_widgets_from_layout(elem):
        """Recursively extract widgets from layout items."""
        for child in elem:
            if child.tag == 'item':
                for subchild in child:
                    if subchild.tag == 'widget':
                        widgets.append(subchild)
                    elif subchild.tag == 'layout':
                        extract_widgets_from_layout(subchild)
            elif child.tag == 'widget':
                widgets.append(child)
            elif child.tag == 'layout':
                extract_widgets_from_layout(child)

    extract_widgets_from_layout(layout)
    print(f"Extracted {len(widgets)} widgets from layout")

    # Remove the layout
    analysis_tab.remove(layout)
    print("Removed layout from Analysis tab")

    # Add geometry back to widgets and add them directly to the tab
    widget_positions = {
        'layoutWidget': (10, 0, 708, 30),  # Browse button section
        'groupBox_2': (10, 30, 161, 71),   # Channel Selection
        'groupBox': (180, 30, 345, 71),    # Filtering & Preprocessing
        'PeakDetection': (530, 30, 270, 71),  # Peak Detection
        'MainPlot': (9, 109, 1281, 441)    # Main plot area
    }

    for widget in widgets:
        name = widget.get('name')

        # Only add top-level widgets (not nested ones)
        if name in widget_positions:
            x, y, w, h = widget_positions[name]

            # Add geometry property
            geom_prop = ET.Element('property', attrib={'name': 'geometry'})
            rect = ET.SubElement(geom_prop, 'rect')
            ET.SubElement(rect, 'x').text = str(x)
            ET.SubElement(rect, 'y').text = str(y)
            ET.SubElement(rect, 'width').text = str(w)
            ET.SubElement(rect, 'height').text = str(h)

            # Insert geometry as first property
            widget.insert(0, geom_prop)

            # Add widget back to tab
            analysis_tab.append(widget)
            print(f"  Restored {name} with geometry ({x}, {y}, {w}, {h})")

    # Pretty print the XML
    ET.indent(tree, space='  ', level=0)

    # Write back
    tree.write(ui_file, encoding='UTF-8', xml_declaration=True)
    print(f"\nRestored: {ui_file}")

if __name__ == "__main__":
    ui_dir = Path(__file__).parent / "ui"
    ui_file = ui_dir / "pleth_app_layout_02_horizontal.ui"

    restore_analysis_tab(ui_file)
    print("\nDone! Analysis tab restored to absolute positioning.")
