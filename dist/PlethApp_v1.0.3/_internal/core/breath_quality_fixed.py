# core/breath_quality_fixed.py
"""
Breath quality assessment and error detection for visual highlighting.

This module analyzes the quality of detected breath events and identifies
problematic cycles that should be highlighted for user attention.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional


def assess_breath_quality(t: np.ndarray, y: np.ndarray, peaks: np.ndarray,
                         onsets: np.ndarray, offsets: np.ndarray,
                         expmins: np.ndarray, expoffs: np.ndarray,
                         sr_hz: float) -> Dict[str, np.ndarray]:
    """
    Assess the quality of breath event detection and identify problematic cycles.

    Returns:
        Dict with keys:
        - 'error_mask': Boolean array (length N) where True indicates problematic regions
        - 'warning_mask': Boolean array (length N) where True indicates questionable regions
        - 'cycle_errors': List of error descriptions for each breath cycle
    """
    N = len(y)
    error_mask = np.zeros(N, dtype=bool)
    warning_mask = np.zeros(N, dtype=bool)
    cycle_errors = []

    if peaks is None or len(peaks) == 0:
        return {
            'error_mask': error_mask,
            'warning_mask': warning_mask,
            'cycle_errors': []
        }

    # Analyze each breath cycle
    for i in range(len(peaks)):
        try:
            peak_idx = int(peaks[i])  # Convert to Python int immediately
            errors = []
            warnings = []

            # Determine cycle boundaries for highlighting (peak-to-peak or beginning/end)
            if i == 0:
                # First peak: start from beginning of trace
                cycle_start = 0
            else:
                # Middle peaks: start halfway between previous peak and this peak
                prev_peak = int(peaks[i-1])
                cycle_start = max(0, (prev_peak + peak_idx) // 2)

            if i == len(peaks) - 1:
                # Last peak: go to end of trace
                cycle_end = N
            else:
                # Middle peaks: end halfway between this peak and next peak
                next_peak = int(peaks[i+1])
                cycle_end = min(N, (peak_idx + next_peak) // 2)

            # Check 1: Missing breath events
            has_onset = onsets is not None and len(onsets) > i
            has_offset = offsets is not None and len(offsets) > i
            has_expmin = expmins is not None and len(expmins) > i
            has_expoff = expoffs is not None and len(expoffs) > i

            if not has_onset:
                errors.append("Missing onset")
            if not has_offset:
                errors.append("Missing offset")
            if not has_expmin:
                warnings.append("Missing expiratory minimum")
            if not has_expoff:
                warnings.append("Missing expiratory offset")

            # Check 2: Event ordering (only if we have the events)
            if has_onset and has_offset:
                onset_idx = int(onsets[i])
                offset_idx = int(offsets[i])

                if onset_idx >= peak_idx:
                    errors.append("Onset after peak")
                if offset_idx <= peak_idx:
                    errors.append("Offset before peak")
                if onset_idx >= offset_idx:
                    errors.append("Onset after offset")

            # Check 3: Reasonable cycle timing
            if has_onset and i + 1 < len(onsets):
                onset_this = int(onsets[i])
                onset_next = int(onsets[i + 1])
                cycle_duration = t[onset_next] - t[onset_this]
                if cycle_duration < 0.5:  # Less than 0.5 seconds
                    warnings.append(f"Very short cycle ({cycle_duration:.2f}s)")
                elif cycle_duration > 10.0:  # More than 10 seconds
                    warnings.append(f"Very long cycle ({cycle_duration:.2f}s)")

            # Check 4: Ti/Te ratio reasonableness
            if has_onset and has_offset and i + 1 < len(onsets):
                onset_this = int(onsets[i])
                offset_this = int(offsets[i])
                onset_next = int(onsets[i + 1])

                ti = t[offset_this] - t[onset_this]
                te = t[onset_next] - t[offset_this]

                if ti <= 0:
                    errors.append("Invalid Ti (≤0)")
                elif ti < 0.1:
                    warnings.append(f"Very short Ti ({ti:.2f}s)")
                elif ti > 5.0:
                    warnings.append(f"Very long Ti ({ti:.2f}s)")

                if te <= 0:
                    errors.append("Invalid Te (≤0)")
                elif te < 0.2:
                    warnings.append(f"Very short Te ({te:.2f}s)")
                elif te > 8.0:
                    warnings.append(f"Very long Te ({te:.2f}s)")

            # Check 5: Peak detection at trace edges (often problematic)
            edge_threshold = int(0.5 * sr_hz)  # 0.5 seconds from edges
            if peak_idx < edge_threshold:
                warnings.append("Peak near start of trace")
            elif peak_idx > N - edge_threshold:
                warnings.append("Peak near end of trace")

            # Store cycle assessment
            cycle_info = {
                'cycle_idx': i,
                'errors': errors,
                'warnings': warnings,
                'severity': 'error' if errors else ('warning' if warnings else 'good'),
                'cycle_start': cycle_start,
                'cycle_end': cycle_end
            }
            cycle_errors.append(cycle_info)

            # Mark regions for highlighting using peak-to-peak boundaries
            if errors:
                error_mask[cycle_start:cycle_end] = True
            elif warnings:
                warning_mask[cycle_start:cycle_end] = True

        except Exception as cycle_error:
            print(f"Error processing breath cycle {i}: {cycle_error}")
            # Mark this peak region as error anyway
            if i == 0:
                cycle_start = 0
            else:
                cycle_start = max(0, (int(peaks[i-1]) + int(peaks[i])) // 2)

            if i == len(peaks) - 1:
                cycle_end = N
            else:
                cycle_end = min(N, (int(peaks[i]) + int(peaks[i+1])) // 2)

            error_mask[cycle_start:cycle_end] = True

            cycle_info = {
                'cycle_idx': i,
                'errors': [f"Processing error: {cycle_error}"],
                'warnings': [],
                'severity': 'error',
                'cycle_start': cycle_start,
                'cycle_end': cycle_end
            }
            cycle_errors.append(cycle_info)

    return {
        'error_mask': error_mask,
        'warning_mask': warning_mask,
        'cycle_errors': cycle_errors
    }


def create_quality_report(cycle_errors: List[Dict]) -> str:
    """Create a human-readable quality report."""
    if not cycle_errors:
        return "No breath cycles detected."

    total_cycles = len(cycle_errors)
    error_cycles = sum(1 for c in cycle_errors if c['severity'] == 'error')
    warning_cycles = sum(1 for c in cycle_errors if c['severity'] == 'warning')
    good_cycles = total_cycles - error_cycles - warning_cycles

    report = f"Breath Quality Report:\n"
    report += f"  Total cycles: {total_cycles}\n"
    report += f"  Good quality: {good_cycles} ({100*good_cycles/total_cycles:.1f}%)\n"
    report += f"  With warnings: {warning_cycles} ({100*warning_cycles/total_cycles:.1f}%)\n"
    report += f"  With errors: {error_cycles} ({100*error_cycles/total_cycles:.1f}%)\n"

    if error_cycles > 0:
        report += f"\nError Details:\n"
        for i, cycle in enumerate(cycle_errors):
            if cycle['severity'] == 'error':
                report += f"  Cycle {i+1}: {', '.join(cycle['errors'])}\n"

    return report