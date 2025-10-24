from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
import re


def load_data_file(path: Path, progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray]:
    """
    Load data file - dispatches to appropriate loader based on file extension

    Supported formats:
    - .abf: Axon Binary Format (pyabf)
    - .smrx: Son64 format (CED Spike2) via Python 3.9 bridge
    - .edf: European Data Format (pyedflib)

    Returns (sr_hz, sweeps_by_channel, channel_names, t)
    sweeps_by_channel[channel] -> 2D array (n_samples, n_sweeps)

    progress_callback: optional function(current, total, message) for progress updates
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.abf':
        return load_abf(path, progress_callback)
    elif suffix == '.smrx':
        from core.io.son64_loader import load_son64
        return load_son64(str(path), progress_callback)
    elif suffix == '.edf':
        from core.io.edf_loader import load_edf
        return load_edf(path, progress_callback)
    else:
        raise ValueError(f"Unsupported file format: {suffix}\nSupported formats: .abf, .smrx, .edf")


def load_abf(path: Path, progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray]:
    """
    Returns (sr_hz, sweeps_by_channel, channel_names, t)
    sweeps_by_channel[channel] -> 2D array (n_samples, n_sweeps)

    progress_callback: optional function(current, total, message) for progress updates
    """
    try:
        import pyabf
    except ImportError:
        raise RuntimeError("pyabf not installed. pip install pyabf")

    if progress_callback:
        progress_callback(0, 100, "Opening ABF file...")

    abf = pyabf.ABF(str(path))
    sr_hz = float(abf.dataRate)
    chan_names = [abf.adcNames[c] for c in range(abf.channelCount)]

    if progress_callback:
        progress_callback(10, 100, "Reading channel data...")

    # Collect sweeps into arrays per channel
    sweeps_by_channel: Dict[str, np.ndarray] = {}
    n_sweeps = abf.sweepCount
    n_samples = abf.sweepPointCount
    total_ops = abf.channelCount * n_sweeps

    op_count = 0
    for ch in range(abf.channelCount):
        abf.setSweep(0, channel=ch)
        M = np.empty((n_samples, n_sweeps), dtype=float)
        for s in range(n_sweeps):
            abf.setSweep(sweepNumber=s, channel=ch)
            M[:, s] = abf.sweepY

            op_count += 1
            if progress_callback and op_count % 5 == 0:  # Update every 5 sweeps
                pct = 10 + int(80 * op_count / total_ops)
                progress_callback(pct, 100, f"Loading channel {ch+1}/{abf.channelCount}, sweep {s+1}/{n_sweeps}...")

        sweeps_by_channel[chan_names[ch]] = M

    if progress_callback:
        progress_callback(95, 100, "Finalizing...")

    # time vector for one sweep
    t = np.arange(n_samples, dtype=float) / sr_hz

    if progress_callback:
        progress_callback(100, 100, "Complete")

    return sr_hz, sweeps_by_channel, chan_names, t


def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Extract date from ABF filename (format: YYYYMMDD####.abf or similar)
    Returns date string (YYYYMMDD) or None if not found
    """
    # Try to find YYYYMMDD pattern in filename
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        # Validate it looks like a valid date
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
            return date_str
    return None


def validate_files_for_concatenation(file_paths: List[Path]) -> Tuple[bool, List[str]]:
    """
    Validate that multiple files can be safely concatenated.

    Checks:
    - All files are same type (.abf or .smrx)
    - All files have same number of channels
    - All files have same channel names
    - All files have same sample rate
    - (Optional warning) Files have same date in filename

    Returns (valid, warnings_or_errors)
    warnings_or_errors: List of warning/error messages
    """
    errors = []
    warnings = []

    # Check all files exist
    for path in file_paths:
        if not path.exists():
            errors.append(f"File not found: {path.name}")

    if errors:
        return False, errors

    # Check all files are same type
    extensions = set(path.suffix.lower() for path in file_paths)
    if len(extensions) > 1:
        errors.append(f"Mixed file types detected: {', '.join(extensions)}. All files must be same type (.abf, .smrx, or .edf).")
        return False, errors

    file_type = extensions.pop()

    # Load metadata from all files (without loading full data)
    file_metadata = []

    try:
        if file_type == '.abf':
            import pyabf
            for path in file_paths:
                abf = pyabf.ABF(str(path))
                metadata = {
                    'path': path,
                    'n_channels': abf.channelCount,
                    'channel_names': [abf.adcNames[c] for c in range(abf.channelCount)],
                    'sample_rate': float(abf.dataRate),
                    'n_sweeps': abf.sweepCount,
                    'n_samples': abf.sweepPointCount,
                }
                file_metadata.append(metadata)
        elif file_type == '.smrx':
            # For SMRX files, we'd need to partially load them to get metadata
            # For now, we'll handle this more gracefully in the concatenation function
            errors.append("Multi-file concatenation for .smrx files not yet implemented.")
            return False, errors
        elif file_type == '.edf':
            # For EDF files, we'd need to partially load them to get metadata
            # For now, we'll handle this more gracefully in the concatenation function
            errors.append("Multi-file concatenation for .edf files not yet implemented.")
            return False, errors
        else:
            errors.append(f"Unsupported file type: {file_type}")
            return False, errors

    except Exception as e:
        errors.append(f"Error reading file metadata: {str(e)}")
        return False, errors

    # Compare metadata across files
    first = file_metadata[0]

    for i, meta in enumerate(file_metadata[1:], start=1):
        # Check number of channels
        if meta['n_channels'] != first['n_channels']:
            errors.append(
                f"File {i+1} ({meta['path'].name}) has {meta['n_channels']} channels, "
                f"but file 1 ({first['path'].name}) has {first['n_channels']} channels."
            )

        # Check channel names
        if meta['channel_names'] != first['channel_names']:
            errors.append(
                f"File {i+1} ({meta['path'].name}) has different channel names: "
                f"{meta['channel_names']} vs {first['channel_names']}"
            )

        # Check sample rate (allow small floating point differences)
        if abs(meta['sample_rate'] - first['sample_rate']) > 0.1:
            errors.append(
                f"File {i+1} ({meta['path'].name}) has sample rate {meta['sample_rate']} Hz, "
                f"but file 1 ({first['path'].name}) has {first['sample_rate']} Hz."
            )

        # Check sweep length (different lengths will be padded with NaN)
        if meta['n_samples'] != first['n_samples']:
            warnings.append(
                f"File {i+1} ({meta['path'].name}) has {meta['n_samples']} samples per sweep, "
                f"but file 1 ({first['path'].name}) has {first['n_samples']} samples. "
                f"Shorter sweeps will be padded with NaN values to match the longest sweep."
            )

    # Check date consistency (warning only)
    dates = []
    for meta in file_metadata:
        date = extract_date_from_filename(meta['path'].name)
        if date:
            dates.append(date)

    if len(dates) == len(file_metadata) and len(set(dates)) > 1:
        warnings.append(
            f"Files have different dates in filenames: {', '.join(set(dates))}. "
            f"This may indicate data from different recording sessions."
        )

    if errors:
        return False, errors
    elif warnings:
        return True, warnings
    else:
        return True, []


def load_and_concatenate_abf_files(file_paths: List[Path], progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray, List[Dict]]:
    """
    Load multiple ABF files and concatenate their sweeps.

    Returns (sr_hz, sweeps_by_channel, channel_names, t, file_info)
    sweeps_by_channel[channel] -> 2D array (n_samples, total_sweeps)
    file_info: List of dicts with 'path', 'sweep_start', 'sweep_end' for each file

    progress_callback: optional function(current, total, message) for progress updates
    """
    if not file_paths:
        raise ValueError("No files provided")

    if len(file_paths) == 1:
        # Just load single file normally
        sr, sweeps, channels, t = load_abf(file_paths[0], progress_callback)
        file_info = [{
            'path': file_paths[0],
            'sweep_start': 0,
            'sweep_end': next(iter(sweeps.values())).shape[1] - 1
        }]
        return sr, sweeps, channels, t, file_info

    # Validate files first
    valid, messages = validate_files_for_concatenation(file_paths)
    if not valid:
        raise ValueError("File validation failed:\n" + "\n".join(messages))

    # Load each file
    all_file_data = []
    total_files = len(file_paths)

    for file_idx, path in enumerate(file_paths):
        if progress_callback:
            pct = int(100 * file_idx / total_files)
            progress_callback(pct, 100, f"Loading file {file_idx+1}/{total_files}: {path.name}...")

        sr, sweeps, channels, t = load_abf(path, progress_callback=None)  # Disable internal progress for cleaner display
        all_file_data.append({
            'path': path,
            'sr': sr,
            'sweeps': sweeps,
            'channels': channels,
            't': t
        })

    # Use metadata from first file
    sr_hz = all_file_data[0]['sr']
    channel_names = all_file_data[0]['channels']

    # Find maximum sweep length across all files by checking actual sweep data
    # (not just 't' array, to handle any edge cases)
    max_samples = max(
        next(iter(file_data['sweeps'].values())).shape[0]
        for file_data in all_file_data
    )

    # Create time vector for longest sweep
    t = np.arange(max_samples, dtype=float) / sr_hz

    # Track which files were padded
    padded_files = []

    # Concatenate sweeps for each channel, padding shorter sweeps with NaN
    sweeps_by_channel = {}
    file_info = []
    current_sweep_idx = 0

    for channel in channel_names:
        channel_sweeps = []
        for file_idx, file_data in enumerate(all_file_data):
            file_sweep_data = file_data['sweeps'][channel]
            n_samples_this_file = file_sweep_data.shape[0]
            n_sweeps_this_file = file_sweep_data.shape[1]

            if n_samples_this_file < max_samples:
                # Pad with NaN to match longest sweep
                padding_size = max_samples - n_samples_this_file
                padding = np.full((padding_size, n_sweeps_this_file), np.nan)
                padded_sweep_data = np.vstack([file_sweep_data, padding])

                # Verify padding worked correctly
                if padded_sweep_data.shape[0] != max_samples:
                    raise ValueError(
                        f"Padding failed for file {file_data['path'].name}: "
                        f"expected {max_samples} samples, got {padded_sweep_data.shape[0]}"
                    )

                channel_sweeps.append(padded_sweep_data)

                # Track that this file was padded (only record once per file)
                if channel == channel_names[0]:  # Only record once
                    padded_files.append({
                        'file_idx': file_idx,
                        'path': file_data['path'],
                        'original_samples': n_samples_this_file,
                        'padded_samples': max_samples
                    })
            else:
                # File already at max length or longer
                channel_sweeps.append(file_sweep_data)

        # Verify all arrays have the same shape[0] before concatenating
        shapes = [arr.shape[0] for arr in channel_sweeps]
        if len(set(shapes)) > 1:
            raise ValueError(
                f"Cannot concatenate {channel}: arrays have different lengths {shapes}. "
                f"Expected all to be {max_samples} samples."
            )

        # Concatenate along sweep axis (axis=1)
        sweeps_by_channel[channel] = np.concatenate(channel_sweeps, axis=1)

    # Build file_info with sweep ranges
    current_sweep_idx = 0
    for file_data in all_file_data:
        n_sweeps = next(iter(file_data['sweeps'].values())).shape[1]
        file_info.append({
            'path': file_data['path'],
            'sweep_start': current_sweep_idx,
            'sweep_end': current_sweep_idx + n_sweeps - 1
        })
        current_sweep_idx += n_sweeps

    if progress_callback:
        message = f"Loaded {len(file_paths)} files with {current_sweep_idx} total sweeps"
        if padded_files:
            message += f" ({len(padded_files)} files padded with NaN)"
        progress_callback(100, 100, message)

    # Add padding info to file_info for display
    for pad_info in padded_files:
        file_info[pad_info['file_idx']]['padded'] = True
        file_info[pad_info['file_idx']]['original_samples'] = pad_info['original_samples']
        file_info[pad_info['file_idx']]['padded_samples'] = pad_info['padded_samples']

    return sr_hz, sweeps_by_channel, channel_names, t, file_info
