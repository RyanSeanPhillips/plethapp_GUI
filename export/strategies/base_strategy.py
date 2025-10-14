"""
Base Export Strategy - Abstract interface for experiment-type-specific exports.

All export strategies must implement this interface to ensure consistent behavior
across different experiment types (30Hz stim, Hargreaves, Licking, etc.).
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class ExportStrategy(ABC):
    """Abstract base class for experiment-type-specific export strategies."""

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
    def export_timeseries_csv(self, base_path: Path, progress_callback=None) -> Path:
        """
        Export time series data to CSV.

        The time alignment strategy varies by experiment type:
        - 30Hz Stim: Aligned to stimulus onset (t=0 at stim start)
        - Hargreaves: Aligned to heat onset (event onset)
        - Licking: Breath-by-breath with licking state tags

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved CSV file
        """
        pass

    @abstractmethod
    def export_analysis_csv(self, base_path: Path, progress_callback=None) -> Path:
        """
        Export analysis-specific CSV data.

        Content varies by experiment type:
        - 30Hz Stim: Breath-by-breath metrics with region tags (baseline/stim/post)
        - Hargreaves: Per-event summary (latency, baseline, during, post metrics)
        - Licking: During vs outside licking comparison + histograms

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved CSV file
        """
        pass

    @abstractmethod
    def generate_summary_pdf(self, base_path: Path, progress_callback=None) -> Path:
        """
        Generate visual summary PDF.

        Plot types vary by experiment type:
        - 30Hz Stim: Time series with baseline/stim/post regions
        - Hargreaves: Heatmap aligned to onset, latency distributions
        - Licking: Histograms (during vs outside), window analysis

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved PDF file
        """
        pass

    def export_npz_bundle(self, base_path: Path, progress_callback=None) -> Path:
        """
        Export NPZ bundle with complete analysis state (UNIVERSAL - same for all types).

        This is a "project save file" that stores everything needed to resume editing:
        - Raw channel data
        - Detected peaks and breath events
        - Event channel annotations (bout_annotations)
        - Manual edits and omitted sweeps
        - Filter and detection parameters
        - Downsampled timeseries for fast consolidation

        NOTE: This method is NOT abstract - all strategies use the same NPZ format.
        Only the experiment_type field differs (used by export strategies, not NPZ structure).

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved NPZ file
        """
        # Delegate to ExportManager's existing NPZ export
        # This ensures all experiment types save identical NPZ structure
        return self.export_manager._export_bundle_npz(base_path, progress_callback)

    def export_events_csv(self, base_path: Path, progress_callback=None) -> Path:
        """
        Export event intervals CSV (UNIVERSAL - same for all types).

        Contains:
        - Stimulus timing (on/off)
        - Eupnea/apnea periods
        - Event channel annotations (if present)
        - Sniffing bouts (if marked)

        NOTE: This method is NOT abstract - all strategies use the same event CSV format.

        Args:
            base_path: Base file path (without extension)
            progress_callback: Optional function(progress_pct) for UI updates

        Returns:
            Path to saved CSV file
        """
        # Delegate to ExportManager's existing events export
        return self.export_manager._export_events_csv(base_path, progress_callback)

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

        Returns:
            List of suffixes (e.g., ["_bundle.npz", "_timeseries.csv", "_analysis.csv", "_events.csv", "_summary.pdf"])
        """
        return [
            "_bundle.npz",
            "_timeseries.csv",
            "_analysis.csv",
            "_events.csv",
            "_summary.pdf"
        ]
