# PlethApp Event Handling Analysis - Documentation Index

**Analysis Date:** 2025-10-13
**Issue:** Editing modes (Mark Sniff, Add Peaks, Move Point, etc.) stop working after Y2 axis changes
**Status:** Root cause identified, fix implemented, verification pending

---

## Quick Start

**If you just want to know what went wrong and how to fix it:**
→ Read **[ANALYSIS_SUMMARY.md](./ANALYSIS_SUMMARY.md)** (15 min read)

**If you're debugging and need immediate help:**
→ Use **[TROUBLESHOOTING_GUIDE.md](./TROUBLESHOOTING_GUIDE.md)** (step-by-step diagnostics)

**If you need to understand the complete architecture:**
→ Read **[EVENT_HANDLING_ARCHITECTURE.md](./EVENT_HANDLING_ARCHITECTURE.md)** (comprehensive analysis)

**If you prefer visual diagrams:**
→ See **[EVENT_FLOW_DIAGRAM.md](./EVENT_FLOW_DIAGRAM.md)** (ASCII flow charts)

---

## Document Overview

### 1. EVENT_HANDLING_ARCHITECTURE.md (Primary Technical Document)

**Length:** ~400 lines, 10 major sections
**Audience:** Developers, architects, maintainers
**Purpose:** Complete technical analysis of event handling system

**Contents:**
1. Architecture Overview - Module hierarchy and responsibilities
2. Complete Click Event Flow - Step-by-step event propagation
3. Event Connection Lifecycle - Creation, survival, destruction
4. Y2 Axis Addition Flow - Detailed breakdown of the problem scenario
5. Current Fix Analysis - How the fix works and why
6. Other Redraw Scenarios - Comparison with working cases
7. Root Cause Summary - Fundamental architectural issue
8. Diagnostic Checklist - Verification steps
9. Recommended Next Steps - Short and long-term improvements
10. Conclusion - Final verdict and confidence assessment

**Key Insights:**
- Matplotlib's three-level event system (canvas, axes, application)
- Why canvas connections survive `fig.clear()` but callbacks don't
- Why Y2 axis addition uniquely exposed the bug
- Complete restoration logic explanation

**When to read:** You need to understand WHY the bug exists and HOW the fix works at a deep level.

---

### 2. EVENT_FLOW_DIAGRAM.md (Visual Reference)

**Length:** ~300 lines, 5 major diagrams
**Audience:** Visual learners, troubleshooters, QA testers
**Purpose:** Show event flows visually with ASCII diagrams

**Contents:**
- Diagram 1: Normal Click Event Flow (Working)
- Diagram 2: Connection Lifecycle (Creation to Destruction)
- Diagram 3: Y2 Axis Event Blocking Strategy
- Diagram 4: Bug Scenario Timeline (Before Fix)
- Diagram 5: Comparison Matrix (All Scenarios)

**Key Visuals:**
- Module interaction hierarchy
- Event propagation chain
- Connection survival through `fig.clear()`
- Timeline showing when callback is lost
- Before/after comparison

**When to read:** You want to SEE how events flow rather than just reading descriptions.

---

### 3. ANALYSIS_SUMMARY.md (Executive Summary)

**Length:** ~200 lines, 11 sections
**Audience:** Project managers, quick readers, decision makers
**Purpose:** Understand the issue and fix without deep technical details

**Contents:**
1. Key Findings - Root cause in plain language
2. Why Y2 Exposed the Bug - Comparison table
3. The Implemented Fix - What changed and where
4. Architecture Overview - High-level module diagram
5. Event Flow Simplified - 3 scenarios (normal, broken, fixed)
6. Verification Checklist - How to test
7. Potential Issues - What might still be wrong
8. Comparison with Other Modes - Impact scope
9. Files Modified - Change summary
10. Follow-up Actions - Immediate and long-term
11. Conclusion - Confidence and next steps

**Key Takeaways:**
- Single-page comparison table showing Y2 is the unique failure case
- Simple before/after diagrams
- Clear verification procedure
- Risk assessment

**When to read:** You need a quick understanding without diving into technical details.

---

### 4. TROUBLESHOOTING_GUIDE.md (Practical Reference)

**Length:** ~250 lines, diagnostic workflows
**Audience:** Developers actively debugging, QA testers
**Purpose:** Step-by-step problem diagnosis and fixes

**Contents:**
- Symptom Checklist - What's broken, when does it break
- Quick Diagnostic Steps - 3-step verification (5-30 seconds each)
- Common Issues and Fixes - Issue patterns with solutions
- Emergency Workaround - Band-aid fix if all else fails
- Advanced Debugging - Connection audit and validation scripts
- Contact Points in Code - Exact line numbers to modify
- Success Indicators - How to know it's working

**Key Tools:**
- Copy-paste debug code snippets
- Expected vs actual console output comparisons
- Decision tree for diagnosis
- Python console validation script

**When to read:** You're actively debugging the issue right now and need immediate help.

---

## Problem Summary (TL;DR)

### What Went Wrong?

1. **User activates Mark Sniff mode** → Click callback registered ✅
2. **User selects Y2 metric** → Plot redraws with `fig.clear()` 💥
3. **Callback registration lost** → Clicks stop working ❌

### Why It Happened?

- `fig.clear()` destroys axes but not canvas
- Canvas-level connections (`_cid_button`) survive ✓
- Application-level registrations (`_external_click_cb`) get orphaned ✗
- No automatic restoration mechanism existed

### How It Was Fixed?

Added `_restore_editing_mode_connections()` method that:
1. Explicitly re-registers `_external_click_cb` after every redraw
2. Runs AFTER Y2 axis creation (critical timing)
3. Handles all editing modes (Mark Sniff, Add Peaks, Delete Peaks, Add Sigh, Move Point)

### Verification Status?

**Pending user testing** - Fix is implemented but needs verification with this test sequence:
1. Activate Mark Sniff mode
2. Test click (should work)
3. Select Y2 metric "IF"
4. Test click again (should still work with fix)

---

## Code Locations

### The Fix
- **File:** `plotting/plot_manager.py`
- **Method:** `_restore_editing_mode_connections()` (lines 433-515)
- **Called from:** `_draw_single_panel_plot()` (line ~40)

### The Problem Source
- **File:** `core/plotting.py`
- **Method:** `show_trace_with_spans()` (line 278: `fig.clear()`)
- **Method:** `add_or_update_y2()` (lines 546-598: Y2 axis creation)

### Event Handlers
- **File:** `core/plotting.py`
- **Method:** `_on_button()` (lines 151-173: Click handler)
- **Method:** `set_click_callback()` (line 656-658: Callback registration)

### Editing Modes
- **File:** `editing/editing_modes.py`
- **Method:** `on_mark_sniff_toggled()` (lines 1173-1241: Mode activation)
- **Method:** `_on_plot_click_mark_sniff()` (lines 1242-1308: Click handler)

---

## Key Technical Terms

| Term | Definition | Survival |
|------|-----------|----------|
| **Canvas Connection** | `canvas.mpl_connect()` | ✅ Survives `fig.clear()` |
| **Axes Callback** | `ax.callbacks.connect()` | ❌ Destroyed by `fig.clear()` |
| **Callback Registration** | `_external_click_cb = fn` | ⚠️ No automatic persistence |
| **_cid_button** | Connection ID for button_press_event | Persistent (canvas level) |
| **_external_click_cb** | Application-level callback routing | Orphaned by redraws |
| **fig.clear()** | Destroys all axes, recreates blank figure | Trigger point for bug |

---

## Timeline

| Date | Event |
|------|-------|
| **Pre-Y2 Feature** | Bug existed but rarely triggered (fewer redraws) |
| **Y2 Feature Added** | Bug exposed (frequent redraws with metric toggling) |
| **2025-10-13** | Root cause analysis performed |
| **2025-10-13** | Fix implemented (`_restore_editing_mode_connections`) |
| **TBD** | User verification and testing |

---

## Impact Assessment

### Affected Features
- ✅ Mark Sniff mode (click-and-drag region marking)
- ✅ Add Peaks mode (click to add inspiratory peaks)
- ✅ Delete Peaks mode (click to remove peaks)
- ✅ Add Sigh mode (click to toggle sigh markers)
- ✅ Move Point mode (click to select, drag to move breath events)

### Unaffected Features
- ✅ Sweep navigation (restoration already existed)
- ✅ Filter parameter changes (restoration already existed)
- ✅ Peak detection (restoration already existed)
- ✅ Double-click autoscale (PlotHost internal handler)
- ✅ Scroll-wheel zoom (PlotHost internal handler)
- ✅ Matplotlib toolbar (zoom, pan, home, back, forward)

### Scope
- **Before Fix:** Y2 selection broke ALL editing modes
- **After Fix:** All editing modes should work regardless of Y2 state

---

## Related Issues

### Similar Bugs (Hypothetical)
If this pattern occurs elsewhere in the codebase, look for:
1. Application-level registrations that depend on figure/axes state
2. Callback chains that break during redraws
3. Temporary connections that need restoration after `fig.clear()`

### Prevention Strategy
1. **Defensive restoration:** Always restore after redraws
2. **Query-based resolution:** Don't store callbacks, query mode state on every click
3. **Connection registry:** Track all connections for auditing
4. **Minimize `fig.clear()`:** Use `ax.cla()` where possible

---

## Questions & Answers

### Q: Why didn't sweep navigation break?
**A:** It already had restoration logic from an earlier fix. Y2 selection path was missing it.

### Q: Why do canvas connections survive but callbacks don't?
**A:** Canvas is a PyQt widget that persists. Figure/axes are matplotlib objects destroyed by `clear()`. Callbacks are Python variables with no automatic persistence.

### Q: Could we eliminate the need for restoration?
**A:** Yes, by querying mode state on every click instead of storing callbacks. This would be a better long-term architecture.

### Q: Is there a performance impact?
**A:** No. Restoration runs once per redraw (already expensive). The fix adds negligible overhead.

### Q: Can this break again?
**A:** Yes, if new redraw code paths are added that don't call restoration. Need regression test.

---

## Next Steps for Developer

1. **Read ANALYSIS_SUMMARY.md** (understand the issue)
2. **Verify fix is in place** (check `plotting/plot_manager.py` line ~40)
3. **Run verification test** (from ANALYSIS_SUMMARY.md section 6)
4. **Check console debug output** (from TROUBLESHOOTING_GUIDE.md)
5. **If broken, use TROUBLESHOOTING_GUIDE.md** (diagnostic workflow)
6. **If working, add regression test** (prevent future breaks)
7. **Consider long-term refactor** (EVENT_HANDLING_ARCHITECTURE.md section 9)

---

## Documentation Metrics

| Document | Lines | Sections | Time to Read |
|----------|-------|----------|--------------|
| EVENT_HANDLING_ARCHITECTURE.md | 400+ | 10 | 30-40 min |
| EVENT_FLOW_DIAGRAM.md | 300+ | 5 | 15-20 min |
| ANALYSIS_SUMMARY.md | 200+ | 11 | 10-15 min |
| TROUBLESHOOTING_GUIDE.md | 250+ | 8 | 10-20 min (active use) |
| **TOTAL** | **1200+** | **34** | **~1.5 hours (full read)** |

**Files Analyzed:** 3 major modules (~3000 lines of code)
**Analysis Time:** ~45 minutes
**Documentation Time:** ~60 minutes
**Total Effort:** ~2 hours

---

## License & Usage

These documents are part of the PlethApp project and are provided for internal development use. They may be shared with contributors and maintainers. If you found this analysis helpful, consider:

1. **Adding regression tests** to prevent re-occurrence
2. **Improving the architecture** (see section 9 of main document)
3. **Documenting other complex subsystems** similarly
4. **Creating a developer wiki** with these patterns

---

**Created by:** Claude (Anthropic AI)
**Date:** 2025-10-13
**Analysis Version:** 1.0
**Status:** Awaiting user verification

**For questions or clarifications about this analysis, refer to the relevant section in the detailed documents above.**

---

## Quick Access Links

- 📘 [Full Technical Analysis](./EVENT_HANDLING_ARCHITECTURE.md)
- 📊 [Visual Diagrams](./EVENT_FLOW_DIAGRAM.md)
- 📋 [Executive Summary](./ANALYSIS_SUMMARY.md)
- 🔧 [Troubleshooting Guide](./TROUBLESHOOTING_GUIDE.md)
- 📁 [This Index](./README_EVENT_ANALYSIS.md)

---

**End of Documentation Index**
