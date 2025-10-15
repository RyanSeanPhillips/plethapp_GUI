"""
LickingStrategy - PDF summary generation for licking behavior experiments.

This strategy generates PDFs comparing breathing during vs outside licking bouts,
with histograms and transition dynamics analysis.
"""

from pathlib import Path
from .base_strategy import ExportStrategy


class LickingStrategy(ExportStrategy):
    """PDF summary strategy for licking behavior experiments."""

    def generate_summary_pdf(self, base_path: Path, data_dict: dict, progress_callback=None) -> Path:
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
            data_dict: Dictionary containing processed data from ExportManager
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved PDF file (_summary.pdf)
        """
        pdf_path = base_path.with_name(base_path.name + "_summary.pdf")

        if progress_callback:
            progress_callback(90)

        # TODO Phase 2: Implement licking-specific PDF generation
        # Use matplotlib to create multi-page PDF with:
        # - Histogram comparisons (during vs outside licking)
        # - Transition dynamics aligned to bout onset/offset
        # - Per-bout scatter plots
        # - Statistical comparisons (paired t-tests, effect sizes)

        print(f"[Licking] TODO: Implement bout comparison PDF with histograms and transitions")
        print(f"[Licking] Will export to: {pdf_path}")

        return pdf_path

    def get_strategy_name(self) -> str:
        """Return human-readable name of this strategy."""
        return "Licking Behavior"
