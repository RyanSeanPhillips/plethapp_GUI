"""
LickingStrategy - Export strategy for licking behavior experiments.

This strategy compares breathing metrics during licking bouts vs outside licking,
with window analysis around bout transitions and histogram comparisons.
"""

from pathlib import Path
import numpy as np
from .base_strategy import ExportStrategy


class LickingStrategy(ExportStrategy):
    """Export strategy for licking behavior experiments."""

    def export_timeseries_csv(self, base_path: Path, progress_callback=None) -> Path:
        """
        Export breath-by-breath time series with licking state tags.

        CSV contains:
        - t: Absolute time (s)
        - breath_num: Breath index
        - licking_state: Boolean (1 = during licking bout, 0 = outside)
        - bout_id: ID of current licking bout (NaN if outside)
        - time_since_bout_onset: Time since licking bout started (s, NaN if outside)
        - time_until_bout_offset: Time until licking bout ends (s, NaN if outside)

        For each metric (if, amp_insp, ti, te, etc.):
        - {metric}: Raw value
        - {metric}_norm: Normalized to session baseline (mean of all breaths outside licking)

        This format enables:
        - Plotting metrics vs time_since_bout_onset to see transition dynamics
        - Filtering by licking_state for during/outside comparison

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved CSV file (_licking_timeseries.csv)
        """
        csv_path = base_path.with_name(base_path.name + "_licking_timeseries.csv")

        if progress_callback:
            progress_callback(30)

        # TODO: Extract and implement breath-by-breath export with licking state
        # Algorithm:
        # 1. For each breath, determine if it occurred during a licking bout
        # 2. Calculate time since bout onset/until offset
        # 3. Normalize metrics to session baseline (mean outside licking)
        # 4. Export to CSV with licking state tags

        print(f"[Licking] TODO: Implement breath-by-breath timeseries with licking state")
        print(f"[Licking] Will export to: {csv_path}")

        return csv_path

    def export_analysis_csv(self, base_path: Path, progress_callback=None) -> Path:
        """
        Export window analysis CSV comparing during vs outside licking.

        CSV contains two main sections:

        **Section 1: Per-Bout Summary** (one row per licking bout)
        - bout_id: Bout identifier
        - sweep: Sweep index
        - onset_time: Bout onset (s)
        - offset_time: Bout offset (s)
        - duration: Bout duration (s)
        - n_breaths: Number of breaths during bout

        For each metric:
        - during_{metric}_mean: Mean during bout
        - during_{metric}_std: Std during bout
        - baseline_{metric}_mean: Mean in matched baseline window (before onset)
        - delta_{metric}: during_mean - baseline_mean
        - pct_change_{metric}: ((during - baseline) / baseline) * 100

        **Section 2: Transition Analysis** (time-locked to bout onset/offset)
        - time_bin: Time bin relative to onset/offset (e.g., -2 to +5s in 0.5s bins)
        - {metric}_mean: Mean across all bouts at this time bin
        - {metric}_sem: SEM across bouts

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved CSV file (_licking_analysis.csv)
        """
        analysis_path = base_path.with_name(base_path.name + "_licking_analysis.csv")

        if progress_callback:
            progress_callback(60)

        # TODO: Extract and implement window analysis logic
        # Algorithm:
        # 1. For each bout, extract metrics during and in matched baseline window
        # 2. Compute per-bout summary statistics
        # 3. Align all bouts to onset/offset and bin in time windows
        # 4. Export both per-bout and transition analysis to CSV

        print(f"[Licking] TODO: Implement window analysis and transition dynamics")
        print(f"[Licking] Will export to: {analysis_path}")

        return analysis_path

    def generate_summary_pdf(self, base_path: Path, progress_callback=None) -> Path:
        """
        Generate visual summary PDF for licking behavior experiments.

        PDF contains:
        - Page 1: Histogram comparisons (during vs outside licking)
          - 2x3 grid: IF, Amplitude_insp, Ti, Te, Vent_proxy, Area_insp
          - Each plot shows overlapping histograms: blue (outside), orange (during)
          - Statistical annotations (t-test p-values, effect sizes)

        - Page 2: Transition dynamics (time-locked to bout onset)
          - Line plots showing mean ± SEM for each metric
          - Vertical line at t=0 (bout onset)
          - Shaded region indicating typical bout duration
          - Separate plots for IF, amplitude, Ti, Te

        - Page 3: Transition dynamics (time-locked to bout offset)
          - Same as page 2 but aligned to bout offset
          - Shows recovery dynamics after licking stops

        - Page 4: Per-bout scatter plots
          - Each point is one bout
          - X-axis: baseline metric, Y-axis: during metric
          - Unity line (x=y) for reference
          - Color-coded by bout duration

        - Page 5: Statistical summary table
          - Mean ± SD for during vs outside
          - Paired t-test results
          - Effect sizes (Cohen's d)

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved PDF file (_licking_summary.pdf)
        """
        pdf_path = base_path.with_name(base_path.name + "_licking_summary.pdf")

        if progress_callback:
            progress_callback(90)

        # TODO: Extract and implement PDF generation logic
        # Use matplotlib to create multi-page PDF with histograms and transition plots

        print(f"[Licking] TODO: Implement summary PDF with histograms and transitions")
        print(f"[Licking] Will export to: {pdf_path}")

        return pdf_path

    def get_strategy_name(self) -> str:
        """Return human-readable name of this strategy."""
        return "Licking Behavior"

    def get_expected_file_suffixes(self) -> list[str]:
        """
        Return list of file suffixes this strategy will create.

        Returns:
            List of suffixes for licking exports
        """
        return [
            "_bundle.npz",
            "_licking_timeseries.csv",
            "_licking_analysis.csv",
            "_events.csv",
            "_licking_summary.pdf"
        ]
