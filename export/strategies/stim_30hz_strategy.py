"""
Stim30HzStrategy - PDF summary generation for standard 30Hz stimulation experiments.

This strategy generates PDFs with time series aligned to stimulus onset and
histograms comparing baseline/stim/post regions.
"""

from pathlib import Path
from .base_strategy import ExportStrategy


class Stim30HzStrategy(ExportStrategy):
    """PDF summary strategy for standard 30Hz stimulation experiments (default behavior)."""

    def generate_summary_pdf(self, base_path: Path, data_dict: dict, progress_callback=None) -> Path:
        """
        Generate visual summary PDF for 30Hz stimulation experiments.

        PDF contains:
        - Page 1: Time series plots (IF, Amplitude, Ti, Te) with baseline/stim/post shading
          - Mean ± SEM across sweeps
          - Vertical line at t=0 (stimulus onset)
          - Shaded regions for baseline (gray), stim (yellow), post (white)

        - Page 2: Histogram distributions comparing regions
          - 2x3 grid: IF, Amplitude, Ti, Te, Vent_proxy, Area
          - Each plot shows overlapping histograms: all (blue), baseline (gray), stim (orange), post (green)
          - Statistical annotations (mean ± SD for each region)

        - Page 3: Statistical summary tables
          - Mean ± SD for each metric in each region
          - Comparisons: baseline vs stim, baseline vs post

        Args:
            base_path: Base file path (without extension)
            data_dict: Dictionary containing processed data from ExportManager
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved PDF file (_summary.pdf)
        """
        # NOTE: For now, delegate to ExportManager's existing inline PDF logic
        # The actual PDF generation happens in ExportManager._export_all_analyzed_data
        # (lines ~1605-2220 in export_manager.py)
        # TODO Phase 2: Extract PDF generation logic here

        pdf_path = base_path.with_name(base_path.name + "_summary.pdf")

        if progress_callback:
            progress_callback(90)

        print(f"[Stim30Hz] Delegating PDF generation to ExportManager existing logic")
        print(f"[Stim30Hz] Will export to: {pdf_path}")

        return pdf_path

    def get_strategy_name(self) -> str:
        """Return human-readable name of this strategy."""
        return "30Hz Stimulus (Standard)"
