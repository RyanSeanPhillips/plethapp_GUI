"""Add proper layout to Analysis tab so MainPlot expands while controls stay fixed."""
from pathlib import Path
import xml.etree.ElementTree as ET

def fix_analysis_tab_layout(ui_file):
    """
    Convert Analysis tab from absolute positioning to proper QVBoxLayout.
    - Browse button and groupboxes stay fixed at top
    - MainPlot expands to fill available space
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

    # Find all the widgets we need to put in layout
    # These currently have absolute geometry positioning
    widgets_to_layout = []
    for child in list(analysis_tab):
        if child.tag == 'widget':
            widget_name = child.get('name')
            print(f"  Found widget: {widget_name}")
            widgets_to_layout.append(child)

    # Remove all widgets from tab (we'll add them back in layout)
    for widget in widgets_to_layout:
        analysis_tab.remove(widget)

    # Remove geometry properties from each widget (no longer needed with layout)
    for widget in widgets_to_layout:
        for prop in list(widget):
            if prop.tag == 'property' and prop.get('name') == 'geometry':
                widget.remove(prop)
                print(f"  Removed geometry from {widget.get('name')}")

    # Create the main vertical layout
    layout = ET.Element('layout', attrib={'class': 'QVBoxLayout', 'name': 'analysisTabLayout'})

    # Set tight margins and spacing
    margins = ET.SubElement(layout, 'property', attrib={'name': 'spacing'})
    ET.SubElement(margins, 'number').text = '5'

    left_margin = ET.SubElement(layout, 'property', attrib={'name': 'leftMargin'})
    ET.SubElement(left_margin, 'number').text = '10'

    right_margin = ET.SubElement(layout, 'property', attrib={'name': 'rightMargin'})
    ET.SubElement(right_margin, 'number').text = '10'

    top_margin = ET.SubElement(layout, 'property', attrib={'name': 'topMargin'})
    ET.SubElement(top_margin, 'number').text = '5'

    bottom_margin = ET.SubElement(layout, 'property', attrib={'name': 'bottomMargin'})
    ET.SubElement(bottom_margin, 'number').text = '10'

    # Add widgets to layout in order:
    # 1. Browse section (layoutWidget with horizontalLayout_7)
    # 2. Groupboxes row (create horizontal layout)
    # 3. MainPlot (with stretch)

    # Find the widgets by name
    browse_widget = None
    groupbox_2 = None
    groupbox = None
    peak_detection = None
    main_plot = None

    for widget in widgets_to_layout:
        name = widget.get('name')
        if name == 'layoutWidget':  # Browse button section
            browse_widget = widget
        elif name == 'groupBox_2':  # Channel Selection
            groupbox_2 = widget
        elif name == 'groupBox':  # Filtering & Preprocessing
            groupbox = widget
        elif name == 'PeakDetection':
            peak_detection = widget
        elif name == 'MainPlot':
            main_plot = widget

    # Add Browse section (fixed size, no stretch)
    if browse_widget:
        browse_item = ET.SubElement(layout, 'item')
        browse_item.append(browse_widget)
        print("  Added Browse section to layout")

    # Create horizontal layout for the three groupboxes
    groupbox_item = ET.SubElement(layout, 'item')
    groupbox_hlayout = ET.SubElement(groupbox_item, 'layout', attrib={'class': 'QHBoxLayout', 'name': 'groupBoxesLayout'})

    # Add spacing
    gb_spacing = ET.SubElement(groupbox_hlayout, 'property', attrib={'name': 'spacing'})
    ET.SubElement(gb_spacing, 'number').text = '5'

    # Add groupboxes in order
    if groupbox_2:
        gb2_item = ET.SubElement(groupbox_hlayout, 'item')
        gb2_item.append(groupbox_2)
        print("  Added groupBox_2 to layout")

    if groupbox:
        gb_item = ET.SubElement(groupbox_hlayout, 'item')
        gb_item.append(groupbox)
        print("  Added groupBox to layout")

    if peak_detection:
        pd_item = ET.SubElement(groupbox_hlayout, 'item')
        pd_item.append(peak_detection)
        print("  Added PeakDetection to layout")

    # Add horizontal spacer after groupboxes to push them left
    spacer_item = ET.SubElement(groupbox_hlayout, 'item')
    spacer = ET.SubElement(spacer_item, 'spacer', attrib={'name': 'horizontalSpacer'})
    spacer_orient = ET.SubElement(spacer, 'property', attrib={'name': 'orientation'})
    ET.SubElement(spacer_orient, 'enum').text = 'Qt::Horizontal'
    spacer_size = ET.SubElement(spacer, 'property', attrib={'name': 'sizeHint'})
    spacer_size_elem = ET.SubElement(spacer_size, 'size')
    ET.SubElement(spacer_size_elem, 'width').text = '40'
    ET.SubElement(spacer_size_elem, 'height').text = '20'
    print("  Added horizontal spacer")

    # Add MainPlot with vertical stretch so it expands
    if main_plot:
        plot_item = ET.SubElement(layout, 'item')

        # Add stretch property to make MainPlot expand
        stretch_prop = ET.SubElement(plot_item, 'property', attrib={'name': 'stretch'})
        ET.SubElement(stretch_prop, 'number').text = '1'

        plot_item.append(main_plot)
        print("  Added MainPlot to layout with stretch=1")

    # Add the layout to the tab
    analysis_tab.append(layout)
    print("  Added layout to Analysis tab")

    # Write the modified XML back
    tree.write(ui_file, encoding='UTF-8', xml_declaration=True)
    print(f"\nUpdated: {ui_file}")
    print("\nLayout structure:")
    print("  - Browse button (fixed)")
    print("  - Groupboxes row (fixed)")
    print("  - MainPlot (expands)")

if __name__ == "__main__":
    ui_dir = Path(__file__).parent / "ui"
    ui_file = ui_dir / "pleth_app_layout_02_horizontal.ui"

    fix_analysis_tab_layout(ui_file)
    print("\nDone! MainPlot will now expand when window is resized.")
