# PlethApp Performance Profiling Guide

## Quick Start (Fully Automated!)

We've added `@profile` decorators to key export functions. These decorators:
- **Normal use**: Zero overhead (no-op decorator)
- **When profiling**: Track line-by-line timing

## How to Profile Export Operations

### Automated Workflow (Recommended)

**Just run this**:
```bash
run_profiling.bat
```

**What happens automatically**:
1. ✅ App launches with test data loaded
2. ✅ You trigger export (click "View Summary" or "Save Data")
3. ✅ Close the app
4. ✅ **Results automatically saved** to `profiling_results/profile_YYYYMMDD_HHMMSS.txt`
5. ✅ **Results displayed** in terminal for immediate review
6. ✅ Timestamped files kept for comparison

**That's it!** No manual commands needed.

### Manual Workflow (Advanced)

If you want more control:
```bash
# Run with profiling
kernprof -l run_debug.py

# After closing app, generate report
python -m line_profiler run_debug.py.lprof > my_profile.txt
```

## Understanding the Output

The saved file shows line-by-line timing like:
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================
   470                                           @profile
   471                                           def _export_all_analyzed_data(...):
   472         1       1234.0   1234.0      5.2      st = self.window.state
   473         1      15678.0  15678.0     66.3      # ... some slow operation ...
   ...
```

**Look for lines with high `% Time`** - these are your optimization targets!

## Saved Results Location

**Automatic saves go to**: `profiling_results/profile_YYYYMMDD_HHMMSS.txt`

**Example filename**: `profiling_results/profile_20251024_143052.txt`

**Benefits**:
- ✅ Timestamped so you can compare before/after optimizations
- ✅ Text files that I (Claude) can read to help identify bottlenecks
- ✅ Never overwritten - keeps history of all profiling runs
- ✅ Ignored by git (won't clutter your repository)

**To share results with Claude**: Just tell me to read the latest file:
```
"Read the latest profiling results and tell me what's slow"
```

I'll use the Read tool to analyze the file and suggest optimizations!

## Currently Profiled Functions

### export/export_manager.py
- `_export_all_analyzed_data()` - Main export function

**To add more**: Just add `@profile` decorator above any method:
```python
@profile
def _some_method(self):
    # ... code ...
```

## How the @profile Decorator Works

**In export/export_manager.py (lines 20-27)**:
```python
# Enable line profiling when running with kernprof -l
# Otherwise, @profile decorator is a no-op (zero overhead)
try:
    profile  # Check if already defined by kernprof
except NameError:
    def profile(func):
        """No-op decorator when not profiling."""
        return func
```

**When running normally**:
- `profile` is not defined
- The `except` block creates a no-op decorator
- `@profile` just returns the function unchanged
- **Zero overhead!**

**When running with `kernprof -l`**:
- `kernprof` defines `profile` builtin
- The `try` block succeeds
- `@profile` activates line-by-line tracking
- Adds overhead (10-100× slower)

## Interpretation Guide

### What to Look For

1. **% Time column**: Focus on lines with >10% of total time
2. **Hits column**: Lines called many times (loops)
3. **Time column**: Absolute time (microseconds)

### Common Patterns

**Hot loops** (high % Time, high Hits):
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================
   880      1000      50000.0     50.0     45.2      for i in range(n):
   881      1000      45000.0     45.0     40.7          result[i] = expensive_op(data[i])
```
→ **Vectorize this loop!**

**Slow single operation** (high % Time, low Hits):
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================
   920         1      80000.0  80000.0     72.3      df.to_csv(huge_file.csv)
```
→ **IO-bound, may not be optimizable**

**Numpy operation** (low % Time despite complexity):
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
================================================================
   950         1       1200.0   1200.0      1.1      result = np.mean(data, axis=0)
```
→ **Already optimized, leave as-is**

## Next Steps After Profiling

1. **Identify bottlenecks**: Lines with >10% of total time
2. **Check our vectorization plan**: See PERFORMANCE_OPTIMIZATION_PLAN.md
3. **Implement optimizations**: Start with highest % Time operations
4. **Re-profile**: Verify speedup after changes

## Tips

- **Profile with representative data**: Use files with 10+ sweeps, typical size
- **Don't profile during development**: Only when measuring performance
- **Profile one operation at a time**: Export, PDF generation, etc.
- **Compare before/after**: Run profiler before and after optimization

## Troubleshooting

**Error: "NameError: name 'profile' is not defined"**
→ Make sure you added the try/except block (lines 20-27)

**No output file (.lprof)**
→ Check that you used `kernprof -l` (with the `-l` flag)

**App is super slow when profiling**
→ This is normal! line_profiler adds 10-100× overhead

**Want to profile multiple functions**
→ Add `@profile` to each function you want to track

---

## Related Documentation
- **PERFORMANCE_OPTIMIZATION_PLAN.md**: Detailed vectorization strategy
- **ARCHITECTURE.md**: Performance optimization section
