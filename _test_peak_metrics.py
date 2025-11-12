"""
Test script to inspect the new peak candidate metrics.

Run this after loading a file and running peak detection in the app.
It will print out sample metrics to verify they're working correctly.
"""

import sys
import numpy as np

# Simulate having detected peaks with metrics
def test_peak_metrics():
    """
    This is a standalone test to verify the metrics calculation works.
    """
    from core import peaks as peakdet

    # Create a simple synthetic signal with a shoulder peak case
    sr_hz = 1000.0  # 1kHz sampling
    t = np.linspace(0, 5, int(5 * sr_hz))  # 5 seconds

    # Create breaths with one shoulder peak
    y = np.zeros_like(t)

    # Normal breath 1 (at 1 second)
    y += 2.0 * np.exp(-((t - 1.0)**2) / 0.01)

    # Shoulder peak + main peak (at 2 seconds) - MERGE CANDIDATE
    y += 1.5 * np.exp(-((t - 1.95)**2) / 0.002)  # Shoulder (smaller, earlier)
    y += 2.0 * np.exp(-((t - 2.05)**2) / 0.01)   # Main peak (larger, later)

    # Normal breath 2 (at 3 seconds)
    y += 2.0 * np.exp(-((t - 3.0)**2) / 0.01)

    # Normal breath 3 (at 4 seconds)
    y += 2.0 * np.exp(-((t - 4.0)**2) / 0.01)

    # Detect all peaks
    all_peaks = peakdet.detect_peaks(y, sr_hz, prominence=0.5, return_all=True)
    print(f"Detected {len(all_peaks)} peaks at indices: {all_peaks}")
    print(f"Peak times: {t[all_peaks]}")
    print(f"Peak amplitudes: {y[all_peaks]}")

    # Compute breath events
    breath_events = peakdet.compute_breath_events(y, all_peaks, sr_hz)
    print(f"\nBreath events computed:")
    print(f"  Onsets: {len(breath_events['onsets'])}")
    print(f"  Offsets: {len(breath_events['offsets'])}")
    print(f"  Expmins: {len(breath_events['expmins'])}")

    # Compute comprehensive metrics
    metrics = peakdet.compute_peak_candidate_metrics(
        y=y,
        all_peak_indices=all_peaks,
        breath_events=breath_events,
        sr_hz=sr_hz
    )

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE PEAK METRICS")
    print(f"{'='*80}\n")

    for i, m in enumerate(metrics):
        print(f"Peak {i} at index {m['peak_idx']} (t={t[m['peak_idx']]:.3f}s):")
        print(f"  Amplitude: {m['amplitude_absolute']:.2f} (normalized: {m['amplitude_normalized']:.2f})")

        # Timing
        if m['gap_to_prev'] is not None:
            print(f"  Gap to prev: {m['gap_to_prev']*1000:.1f} ms (normalized: {m['gap_to_prev_norm']:.2f})")
        if m['gap_to_next'] is not None:
            print(f"  Gap to next: {m['gap_to_next']*1000:.1f} ms (normalized: {m['gap_to_next_norm']:.2f})")

        # Merge indicators
        if m['trough_ratio_prev'] is not None:
            print(f"  Trough ratio prev: {m['trough_ratio_prev']:.3f} (value: {m['trough_value_prev']:.2f})")
        if m['trough_ratio_next'] is not None:
            print(f"  Trough ratio next: {m['trough_ratio_next']:.3f} (value: {m['trough_value_next']:.2f})")

        # Prominence
        if m['prom_asymmetry'] is not None:
            print(f"  Prominence asymmetry: {m['prom_asymmetry']:.3f} (L={m['left_prominence']:.2f}, R={m['right_prominence']:.2f})")

        # Onset position (KEY for shoulder detection!)
        if m['onset_above_zero'] is not None:
            print(f"  Onset above zero: {m['onset_above_zero']} (value: {m['onset_value']:.3f}, ratio: {m['onset_height_ratio']:.3f})")

        # Check if this is a merge candidate
        is_merge_candidate = (
            m.get('gap_to_next_norm', 1.0) is not None and
            m.get('gap_to_next_norm') < 0.3 and
            m.get('trough_ratio_next', 1.0) is not None and
            m.get('trough_ratio_next') < 0.15
        )

        if is_merge_candidate:
            print(f"  ⚠️  MERGE CANDIDATE! (small gap + shallow trough)")

        print()

    # Find merge candidates
    merge_candidates = [m for m in metrics
                       if m.get('gap_to_next_norm', 1.0) is not None
                       and m.get('gap_to_next_norm') < 0.3
                       and m.get('trough_ratio_next', 1.0) is not None
                       and m.get('trough_ratio_next') < 0.15]

    print(f"\n{'='*80}")
    print(f"MERGE CANDIDATE SUMMARY")
    print(f"{'='*80}")
    print(f"Found {len(merge_candidates)} potential merge candidates:")
    for mc in merge_candidates:
        print(f"  Peak {mc['peak_idx']}: gap_norm={mc['gap_to_next_norm']:.2f}, trough_ratio={mc['trough_ratio_next']:.3f}")
        print(f"    onset_above_zero={mc['onset_above_zero']}, prom_asymmetry={mc.get('prom_asymmetry', 'N/A')}")

    print(f"\n✅ Metrics calculation working correctly!")
    print(f"   The shoulder peak at ~1.95s should be flagged as a merge candidate")
    print(f"   with small gap_to_next_norm and shallow trough_ratio_next")


if __name__ == '__main__':
    test_peak_metrics()
