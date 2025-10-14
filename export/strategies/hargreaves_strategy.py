"""
HargreavesStrategy - Export strategy for Hargreaves thermal sensitivity experiments.

This strategy aligns all metrics to heat onset (t=0 at heat application) and
withdrawal (paw withdrawal latency), exporting data aligned to thermal events.
"""

from pathlib import Path
import numpy as np
from .base_strategy import ExportStrategy


class HargreavesStrategy(ExportStrategy):
    """Export strategy for Hargreaves thermal sensitivity experiments."""

    def export_timeseries_csv(self, base_path: Path, progress_callback=None) -> Path:
        """
        Export time series CSV aligned to heat onset.

        CSV contains:
        - t: Time relative to heat onset (s), t=0 at heat application
        - For each metric (if, amp_insp, ti, te, etc.):
          - Optional per-sweep columns (if enabled)
          - {metric}_mean: Mean across trials
          - {metric}_sem: Standard error of mean
          - {metric}_norm_mean: Normalized to per-trial baseline
          - {metric}_norm_sem: SEM of normalized

        Regions:
        - Baseline: t < 0
        - During heat: 0 <= t < withdrawal_time
        - Post-withdrawal: t >= withdrawal_time

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved CSV file (_hargreaves_timeseries.csv)
        """
        st = self.state

        # Get event annotations (bout_annotations contains heat onset/offset times)
        # Each event bout represents one trial:
        #   - start_time: heat onset
        #   - end_time: paw withdrawal

        csv_path = base_path.with_name(base_path.name + "_hargreaves_timeseries.csv")

        if progress_callback:
            progress_callback(30)

        # TODO: Extract and implement timeseries export logic
        # Algorithm:
        # 1. For each sweep, identify event bouts (heat onset/withdrawal)
        # 2. Align time to heat onset (t=0)
        # 3. Create time windows: baseline (-5 to 0s), during (0 to withdrawal), post (withdrawal to +10s)
        # 4. Interpolate breath metrics to common time grid
        # 5. Compute mean/SEM across trials at each timepoint
        # 6. Export to CSV with columns: t, if_mean, if_sem, amp_mean, amp_sem, etc.

        print(f"[Hargreaves] TODO: Implement timeseries export aligned to heat onset")
        print(f"[Hargreaves] Will export to: {csv_path}")

        return csv_path

    def export_analysis_csv(self, base_path: Path, progress_callback=None) -> Path:
        """
        Export per-trial analysis CSV with latency and response metrics.

        CSV contains per-trial summary (wide layout):

        Columns:
        - trial: Trial number
        - sweep: Sweep index containing this trial
        - heat_onset_time: Absolute time of heat onset (s)
        - withdrawal_time: Absolute time of paw withdrawal (s)
        - latency: Withdrawal latency (withdrawal_time - heat_onset_time)

        Baseline metrics (mean during baseline period, -5 to 0s):
        - baseline_if, baseline_amp_insp, baseline_ti, baseline_te, etc.

        During heat metrics (mean from heat onset to withdrawal):
        - during_if, during_amp_insp, during_ti, during_te, etc.

        Post-withdrawal metrics (mean from withdrawal to +10s):
        - post_if, post_amp_insp, post_ti, post_te, etc.

        Response metrics (during vs baseline comparison):
        - delta_if: during_if - baseline_if
        - pct_change_if: ((during_if - baseline_if) / baseline_if) * 100
        - ... (same for all metrics)

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved CSV file (_hargreaves_trials.csv)
        """
        trials_path = base_path.with_name(base_path.name + "_hargreaves_trials.csv")

        if progress_callback:
            progress_callback(60)

        # TODO: Extract and implement trial-by-trial export logic
        # Algorithm:
        # 1. For each event bout (trial), extract baseline/during/post regions
        # 2. Compute mean metrics in each region
        # 3. Calculate latency and response metrics
        # 4. Export to CSV with one row per trial

        print(f"[Hargreaves] TODO: Implement per-trial analysis export")
        print(f"[Hargreaves] Will export to: {trials_path}")

        return trials_path

    def generate_summary_pdf(self, base_path: Path, progress_callback=None) -> Path:
        """
        Generate visual summary PDF for Hargreaves experiments.

        PDF contains:
        - Page 1: Trial-averaged time series aligned to heat onset
          - Plots: IF, Amplitude, Ti, Te with shading for baseline/during/post
          - Vertical line at t=0 (heat onset)
          - Vertical line at mean withdrawal latency
          - Error bands (SEM)

        - Page 2: Heatmaps aligned to heat onset
          - Each row is one trial
          - Color represents metric value (IF, amplitude, etc.)
          - Vertical line at mean withdrawal latency

        - Page 3: Response distributions
          - Histograms of latency, delta_if, delta_amp, etc.
          - Baseline vs during comparison boxplots

        - Page 4: Statistical summary table
          - Per-trial metrics summary
          - Paired t-tests (baseline vs during, baseline vs post)

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved PDF file (_hargreaves_summary.pdf)
        """
        pdf_path = base_path.with_name(base_path.name + "_hargreaves_summary.pdf")

        if progress_callback:
            progress_callback(90)

        # TODO: Extract and implement PDF generation logic
        # Use matplotlib to create multi-page PDF with trial-averaged plots and heatmaps

        print(f"[Hargreaves] TODO: Implement summary PDF generation")
        print(f"[Hargreaves] Will export to: {pdf_path}")

        return pdf_path

    def get_strategy_name(self) -> str:
        """Return human-readable name of this strategy."""
        return "Hargreaves Thermal Sensitivity"

    def get_expected_file_suffixes(self) -> list[str]:
        """
        Return list of file suffixes this strategy will create.

        Returns:
            List of suffixes for Hargreaves exports
        """
        return [
            "_bundle.npz",
            "_hargreaves_timeseries.csv",
            "_hargreaves_trials.csv",
            "_events.csv",
            "_hargreaves_summary.pdf"
        ]
