"""Reduce font sizes in UI file to make layout more compact."""

def reduce_font_sizes(input_file, output_file):
    """
    Reduce font sizes:
    - 12pt (headers) → 10pt
    - 11pt (labels) → 9pt
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace font sizes
    content = content.replace('<pointsize>12</pointsize>', '<pointsize>10</pointsize>')
    content = content.replace('<pointsize>11</pointsize>', '<pointsize>9</pointsize>')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Reduced fonts: {output_file}")

if __name__ == "__main__":
    from pathlib import Path

    ui_dir = Path(__file__).parent / "ui"
    base_file = ui_dir / "pleth_app_layout_02.ui"

    # Create versions with reduced fonts
    horizontal_file = ui_dir / "pleth_app_layout_02_horizontal.ui"
    tabbed_file = ui_dir / "pleth_app_layout_02_tabbed.ui"

    reduce_font_sizes(base_file, horizontal_file)
    reduce_font_sizes(base_file, tabbed_file)

    print("\nNext steps:")
    print("1. Open Qt Designer")
    print("2. For horizontal layout: Open pleth_app_layout_02_horizontal.ui")
    print("   Follow instructions in ui/HORIZONTAL_COMPACT_GUIDE.md")
    print("3. For tabbed layout: Open pleth_app_layout_02_tabbed.ui")
    print("   Follow instructions in ui/TABBED_LAYOUT_GUIDE.md")
