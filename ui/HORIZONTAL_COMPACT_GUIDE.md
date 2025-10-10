# Horizontal Compact Layout Guide

## Overview
This layout reduces vertical space by organizing controls into multiple columns instead of single rows.

## Font Size Reductions
First, reduce all font sizes:
- **Headers** (File Selection, Channel Selection, etc.): 12pt → 10pt
- **Labels** (Analyze:, Stimulus:, Low Pass, etc.): 11pt → 9pt

## Channel Selection Section
**Current:** 2 items (header, then single row with all controls)
**New:** 3 items in a grid layout

1. Delete `horizontalLayout_8` (contains Analyze/Stimulus/Apply in one row)
2. Add a `QGridLayout` named `channelGrid`
3. Structure:
   ```
   Row 0, Col 0-1: ChannelSelectionLabel (header, span 2 columns)
   Row 1, Col 0: Analyze: [AnalyzeChanSelect dropdown]
   Row 1, Col 1: Stimulus: [StimChanSelect dropdown]
   Row 2, Col 0-1: [ApplyChanPushButton] (centered, span 2 columns)
   ```

## Filtering Section
**Current:** Single horizontal row with 9 widgets
**New:** 2 rows in grid layout

1. Replace `horizontalLayout_9` with `QGridLayout` named `filterGrid`
2. Structure:
   ```
   Row 0, Col 0: [Low Pass checkbox] [LowPassVal]
   Row 0, Col 1: [High Pass checkbox] [HighPassVal]
   Row 0, Col 2: [Filter Order label] [FilterOrderSpin]
   Row 1, Col 0: [Mean Sub checkbox] [MeanSubVal]
   Row 1, Col 1: [Spectral Analysis button]
   Row 1, Col 2: [Invert Signal checkbox]
   ```

## Peak Detection Section
**Current:** Single horizontal row with 7 widgets
**New:** 2 rows in grid layout

1. Replace `horizontalLayout_10` with `QGridLayout` named `peakDetectionGrid`
2. Structure:
   ```
   Row 0, Col 0: [Threshold label] [ThreshVal]
   Row 0, Col 1: [Prominence label] [PeakPromValue]
   Row 0, Col 2: [Min Peak Dist label] [MinPeakDistValue]
   Row 1, Col 0-2: [ApplyPeakFindPushButton] (centered, span 3 columns)
   ```

## Expected Space Savings
- **Channel Section**: No change (already compact)
- **Filtering Section**: 50% reduction (2 rows instead of 1 very long row)
- **Peak Detection Section**: 50% reduction (2 rows instead of 1 long row)
- **Overall**: ~30-40% reduction in total vertical space for these sections

## Implementation Steps in Qt Designer
1. Open `pleth_app_layout_02_horizontal.ui` in Qt Designer
2. Select each horizontal layout mentioned above
3. Right-click → "Morph into" → QGridLayout
4. Drag widgets into grid positions as specified
5. Set row/column spans as needed
6. Adjust alignments (left-align for controls)
7. Save and test

## Tips
- Use "Preview" in Qt Designer to check spacing
- Set grid layout spacing to 5-10px for compact look
- Ensure all widgets maintain their object names (important for Python code)
- Test that the layout doesn't break when window is resized
