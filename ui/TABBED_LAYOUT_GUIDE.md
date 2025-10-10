# Tabbed Interface Layout Guide

## Overview
This layout uses a QTabWidget to organize controls into tabs, showing only one section at a time for maximum space savings.

## Font Size Reductions
Same as horizontal compact:
- **Headers**: 12pt → 10pt
- **Labels**: 11pt → 9pt
- **Tab labels**: 10pt, bold

## Main Structure
Replace the entire control section (File Selection through Peak Detection) with a single QTabWidget.

### Current Structure (in verticalLayout_8):
```
- File Selection label
- horizontalLayout_7 (Browse button + FileNameValue)
- Channel Selection label
- horizontalLayout_8 (Analyze/Stimulus/Apply)
- Filtering label
- horizontalLayout_9 (all filter controls)
- Peak Detection label
- horizontalLayout_10 (all peak detection controls)
```

### New Structure:
```
- QTabWidget (name: controlTabWidget)
  - Tab 1: "Data"
    - Contains File Selection + Channel Selection
  - Tab 2: "Filtering"
    - Contains all filtering controls
  - Tab 3: "Detection"
    - Contains peak detection controls
```

## Tab 1: Data
**Contents:**
1. File Selection section:
   - Label: "File Selection" (10pt, bold)
   - HBoxLayout: [Browse button] [FileNameValue textbox]

2. Channel Selection section:
   - Label: "Channel Selection" (10pt, bold)
   - HBoxLayout: Analyze: [dropdown] Stimulus: [dropdown]
   - Centered Apply button below

## Tab 2: Filtering
**Contents:**
1. Label: "Filtering & Preprocessing" (10pt, bold)
2. Grid layout (2 rows, same as horizontal compact):
   ```
   Row 0: [Low Pass] [High Pass] [Filter Order]
   Row 1: [Mean Sub] [Spectral Analysis] [Invert Signal]
   ```

## Tab 3: Detection
**Contents:**
1. Label: "Peak Detection" (10pt, bold)
2. Grid layout (2 rows):
   ```
   Row 0: [Threshold] [Prominence] [Min Peak Dist]
   Row 1: [Apply button centered]
   ```

## Tab Widget Styling
Add this stylesheet to the QTabWidget:

```css
QTabWidget::pane {
    border: 1px solid #3e3e42;
    background-color: #1e1e1e;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #2d2d30;
    color: #d4d4d4;
    border: 1px solid #3e3e42;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 6px 12px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #007acc;
    color: #ffffff;
}

QTabBar::tab:hover:!selected {
    background-color: #3e3e42;
}
```

## Expected Space Savings
- **Massive reduction**: Only one tab visible at a time
- **Total vertical space**: ~40-50% of current layout (just one tab's content + tab bar)
- **Most compact option** but requires clicking tabs to switch between sections

## Implementation Steps in Qt Designer
1. Open `pleth_app_layout_02_tabbed.ui` in Qt Designer
2. Find `verticalLayout_8` in the object tree
3. Delete all child items (File Selection label through Peak Detection controls)
4. Drag a QTabWidget into `verticalLayout_8`
5. Name it `controlTabWidget`
6. Add 3 tabs (right-click → "Insert Page"):
   - Tab 1 title: "Data"
   - Tab 2 title: "Filtering"
   - Tab 3 title: "Detection"
7. For each tab:
   - Add a QVBoxLayout
   - Drag widgets from the backup file into appropriate tabs
   - Arrange as specified above
8. Apply the stylesheet to the QTabWidget
9. Save and test

## Workflow Considerations
**Pros:**
- Maximum space savings
- Clean, modern interface
- Organizes controls logically

**Cons:**
- Need to switch tabs to access controls
- May slow down workflow if frequently adjusting multiple sections
- Less "at-a-glance" visibility of all settings

## Recommendation
Best for users who:
- Want maximum plot area
- Typically adjust one section at a time
- Are comfortable with tabbed interfaces
