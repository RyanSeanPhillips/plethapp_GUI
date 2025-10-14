"""
Stim30HzStrategy - Export strategy for standard 30Hz stimulation experiments.

This strategy aligns all metrics to stimulus onset (t=0 at stim start) and
exports data in baseline/stim/post regions.
"""

from pathlib import Path
from .base_strategy import ExportStrategy


class Stim30HzStrategy(ExportStrategy):
    """Export strategy for standard 30Hz stimulation experiments (default behavior)."""

    def export_timeseries_csv(self, base_path: Path, progress_callback=None) -> Path:
        """
        Export time series CSV aligned to stimulus onset.

        CSV contains:
        - t: Time relative to stimulus onset (s), t=0 at first stim pulse
        - For each metric (if, amp_insp, ti, te, etc.):
          - Optional per-sweep columns (if enabled)
          - {metric}_mean: Mean across sweeps
          - {metric}_sem: Standard error of mean
          - {metric}_norm_mean: Time-normalized (per-sweep baseline)
          - {metric}_norm_sem: SEM of normalized
          - {metric}_norm_eupnea_mean: Eupnea-normalized (pooled baseline)
          - {metric}_norm_eupnea_sem: SEM of eupnea-normalized

        Regions:
        - Baseline: t < 0
        - Stim: 0 <= t < stim_duration
        - Post: t >= stim_duration

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved CSV file (_timeseries.csv)
        """
        # NOTE: For Phase 1, delegate to ExportManager's existing inline logic
        # In Phase 2, we'll extract this logic into this method

        # For now, this is called from ExportManager._export_all_analyzed_data
        # which handles the timeseries CSV export inline

        csv_path = base_path.with_name(base_path.name + "_timeseries.csv")

        if progress_callback:
            progress_callback(30)

        # The actual export happens in ExportManager._export_all_analyzed_data
        # (lines 710-895 in export_manager.py)
        # TODO: Extract timeseries CSV export logic here in Phase 2

        return csv_path

    def export_analysis_csv(self, base_path: Path, progress_callback=None) -> Path:
        """
        Export breath-by-breath analysis CSV with region tags.

        CSV contains wide layout with breath-by-breath metrics:

        RAW Columns:
        - sweep, breath, t, region (baseline/stim/post)
        - if, amp_insp, amp_exp, ti, te, area_insp, area_exp, vent_proxy
        - is_sigh (1 if sigh, 0 otherwise)

        Then repeated blocks for:
        - TIME-NORMALIZED (_norm suffix): Normalized by per-sweep baseline
        - EUPNEA-NORMALIZED (_norm_eupnea suffix): Normalized by pooled eupneic baseline

        Each normalization has its own set of columns for all/baseline/stim/post regions.

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved CSV file (_breaths.csv)
        """
        # NOTE: For Phase 1, delegate to ExportManager's existing inline logic
        # In Phase 2, we'll extract this logic into this method

        breaths_path = base_path.with_name(base_path.name + "_breaths.csv")

        if progress_callback:
            progress_callback(60)

        # The actual export happens in ExportManager._export_all_analyzed_data
        # (lines 926-1349 in export_manager.py)
        # TODO: Extract breaths CSV export logic here in Phase 2

        return breaths_path

    def generate_summary_pdf(self, base_path: Path, progress_callback=None) -> Path:
        """
        Generate visual summary PDF for 30Hz stimulation experiments.

        PDF contains:
        - Page 1: Time series plots (IF, Amplitude, Ti, Te) with baseline/stim/post shading
        - Page 2: Histogram distributions (all vs baseline vs stim vs post)
        - Page 3: Statistical summary tables

        All plots aligned to stimulus onset (vertical line at t=0).

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved PDF file (_summary.pdf)
        """
        # NOTE: For Phase 1, delegate to ExportManager's existing inline logic
        # In Phase 2, we'll extract this logic into this method

        pdf_path = base_path.with_name(base_path.name + "_summary.pdf")

        if progress_callback:
            progress_callback(90)

        # The actual export happens in ExportManager._export_all_analyzed_data
        # (lines 1561-2174 in export_manager.py)
        # TODO: Extract PDF generation logic here in Phase 2

        return pdf_path

    def get_strategy_name(self) -> str:
        """Return human-readable name of this strategy."""
        return "30Hz Stimulus (Standard)"
