"""
HargreavesStrategy - PDF summary generation for Hargreaves thermal sensitivity experiments.

This strategy generates PDFs with event-aligned heatmaps and latency distributions
for thermal pain experiments.
"""

from pathlib import Path
from .base_strategy import ExportStrategy


class HargreavesStrategy(ExportStrategy):
    """PDF summary strategy for Hargreaves thermal sensitivity experiments."""

    def generate_summary_pdf(self, base_path: Path, data_dict: dict, progress_callback=None) -> Path:
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
            data_dict: Dictionary containing processed data from ExportManager
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved PDF file (_summary.pdf)
        """
        pdf_path = base_path.with_name(base_path.name + "_summary.pdf")

        if progress_callback:
            progress_callback(90)

        # TODO Phase 2: Implement Hargreaves-specific PDF generation
        # Use matplotlib to create multi-page PDF with:
        # - Trial-averaged time series aligned to heat onset (t=0)
        # - Heatmaps (each row = trial, color = metric value)
        # - Latency distributions and response histograms
        # - Statistical comparisons (baseline vs during vs post)

        print(f"[Hargreaves] TODO: Implement event-aligned PDF generation")
        print(f"[Hargreaves] Will export to: {pdf_path}")

        return pdf_path

    def get_strategy_name(self) -> str:
        """Return human-readable name of this strategy."""
        return "Hargreaves Thermal Sensitivity"
