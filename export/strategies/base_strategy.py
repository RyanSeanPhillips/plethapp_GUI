"""
Base Export Strategy - Abstract interface for experiment-type-specific PDF generation.

All export strategies must implement this interface to generate appropriate
summary PDFs for different experiment types (30Hz stim, Hargreaves, Licking, etc.).

NOTE: CSV exports (timeseries, breaths, events) and NPZ bundles are handled
universally by ExportManager. Only PDF visualization varies by experiment type.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class ExportStrategy(ABC):
    """Abstract base class for experiment-type-specific PDF summary generation."""

    def __init__(self, main_window):
        """
        Initialize the export strategy.

        Args:
            main_window: Reference to MainWindow instance for accessing state and UI
        """
        self.window = main_window
        self.state = main_window.state
        self.export_manager = main_window.export_manager

    @abstractmethod
    def generate_summary_pdf(self, base_path: Path, data_dict: dict, progress_callback=None) -> Path:
        """
        Generate experiment-type-specific visual summary PDF.

        Plot types vary by experiment type:
        - 30Hz Stim: Time series with baseline/stim/post regions + region histograms
        - Hargreaves: Event-aligned heatmaps + latency distributions + response metrics
        - Licking: Bout comparison histograms + transition dynamics + state comparisons

        Args:
            base_path: Base file path (without extension)
            data_dict: Dictionary containing all processed data from ExportManager:
                - 't_ds_csv': Time vector for timeseries
                - 'metrics_by_key': Dict of downsampled metric arrays (M, S)
                - 'breath_data': DataFrame or dict with breath-by-breath data
                - 'events_data': DataFrame or dict with event intervals
                - 'kept_sweeps': List of sweep indices included
                - 'experiment_type': Type of experiment ("30hz_stim", "hargreaves", "licking")
                - Additional experiment-specific data as needed
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved PDF file (_summary.pdf)
        """
        pass

    def get_strategy_name(self) -> str:
        """
        Return human-readable name of this strategy.

        Returns:
            Strategy name (e.g., "30Hz Stimulus", "Hargreaves Thermal", "Licking Behavior")
        """
        return self.__class__.__name__.replace("Strategy", "")

    def get_expected_file_suffixes(self) -> list[str]:
        """
        Return list of file suffixes this strategy will create.

        Used for duplicate file checking before export.
        NOTE: All experiment types create the same files, only PDF content differs.

        Returns:
            List of suffixes
        """
        return [
            "_bundle.npz",
            "_timeseries.csv",
            "_breaths.csv",
            "_events.csv",
            "_summary.pdf"
        ]
