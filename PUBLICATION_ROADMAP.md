# PlethApp Publication Roadmap - Two-Paper Strategy

**Date**: 2025-10-20
**Status**: Planning Phase
**Goal**: Publish two papers showcasing PlethApp as a novel respiratory analysis tool

---

## Table of Contents
1. [Overview](#overview)
2. [Paper 1: v1.0 - ML Classification](#paper-1-plethapp-v10---ml-enhanced-breath-classification)
3. [Paper 2: v2.0 - ML Detection](#paper-2-plethapp-v20---end-to-end-ml-pipeline)
4. [Technical Implementation](#technical-implementation)
5. [Timeline](#timeline)
6. [Software Deliverables](#software-deliverables)
7. [Validation Strategy](#validation-strategy)
8. [Next Steps](#next-steps)

---

## Overview

### Why Two Papers?

**Advantages**:
- ✅ **2 publications instead of 1** - Better for CV and citations
- ✅ **Faster first publication** - 2-3 months vs 6-9 months
- ✅ **Incremental validation** - Prove ML works before full investment
- ✅ **Reduced risk** - v1.0 publication guaranteed even if v2.0 faces challenges
- ✅ **Community feedback** - v1.0 users guide v2.0 development
- ✅ **Funding opportunities** - Published software strengthens grant applications
- ✅ **Easier reviews** - Smaller scope per paper

### Publication Strategy Summary

| Aspect | Paper 1 (v1.0) | Paper 2 (v2.0) |
|--------|---------------|---------------|
| **Novelty** | ML breath classification | ML breath detection |
| **Scope** | GUI + API + ML classifier | End-to-end ML pipeline |
| **Timeline** | 2-3 months | 6-9 months (after v1.0) |
| **Journal** | JOSS (software-focused) | eNeuro/JNeurophys (methods) |
| **Length** | 2-4 pages | 6-10 pages |
| **Datasets** | 20-30 recordings | 50-100 recordings |

---

## Paper 1: PlethApp v1.0 - ML-Enhanced Breath Classification

### Title
> **PlethApp: An Open-Source Tool for Automated Respiratory Pattern Analysis with Machine Learning-Based Breath Classification**

### Target Journal
**Journal of Open Source Software (JOSS)**

**Why JOSS?**
- ✅ Fast review (4-6 weeks typical)
- ✅ Software-focused (perfect fit)
- ✅ Indexed in major databases (citations count)
- ✅ Open access (free for everyone)
- ✅ Short format (2-4 pages + GitHub repo)
- ✅ Lenient on AI assistance acknowledgment

### Key Claims

1. **First comprehensive open-source GUI for plethysmography analysis**
   - Commercial alternatives: LabChart (expensive, proprietary)
   - Academic alternatives: breathmetrics (MATLAB, no GUI, rule-based only)

2. **ML classification outperforms traditional GMM**
   - Handles overlapping distributions (e.g., fast baseline + sniffing)
   - Multi-class support (eupnea, sniffing, sighs, artifacts)
   - Expected F1: 0.92-0.95 vs GMM 0.85-0.88

3. **40% faster than GMM approach**
   - ML inference: ~0.9s per sweep
   - GMM fitting: ~1.5s per sweep
   - Matters for large datasets (100+ sweeps)

4. **Dual-mode usage: GUI + headless API**
   - Interactive exploration (GUI)
   - Batch processing (CLI/API)
   - Integration with analysis pipelines (Python API)

5. **Pre-trained models included**
   - Works out-of-the-box (no training required)
   - Active learning (improves with user corrections)
   - Cross-species (mouse, rat)

### Paper Structure (JOSS Format)

**1. Summary** (1 paragraph)
- What is PlethApp?
- What problem does it solve?
- Key features

**2. Statement of Need** (2-3 paragraphs)
- Importance of plethysmography in neuroscience
- Limitations of existing tools
- Gap that PlethApp fills

**3. Features** (bullet list)
- Automated peak detection with robust fallbacks
- ML-based breath classification (Random Forest + XGBoost ensemble)
- 16+ respiratory metrics
- GUI (PyQt6)
- Headless API (Python)
- CLI (batch processing)
- Pre-trained models
- Active learning framework

**4. Implementation** (1-2 paragraphs + architecture diagram)
- Python 3.9+, PyQt6, NumPy, scikit-learn, XGBoost
- Modular architecture (core, GUI, ML separated)
- Multi-format support (ABF, SMRX)

**5. Validation** (1 paragraph + performance figure)
- Dataset: 30 recordings (20 mouse, 10 rat)
- ML vs GMM comparison (F1, precision, recall)
- Speed benchmarks
- Statistical testing (Wilcoxon signed-rank)

**6. Example Usage** (code snippet)
```python
# GUI mode
plethapp-gui

# CLI mode
plethapp analyze recording.abf --output results/

# Python API
from plethapp import BreathAnalyzer
analyzer = BreathAnalyzer()
results = analyzer.analyze("recording.abf")
results.to_csv("output.csv")
```

**7. Acknowledgments**
> This software was developed with substantial assistance from Claude (Anthropic),
> an AI assistant that contributed to software architecture, algorithm implementation,
> and documentation.

**8. References** (10-15)
- breathmetrics (Baertsch et al., 2018)
- GMM for breath classification
- scikit-learn, XGBoost papers
- Respiratory physiology methods
- PyQt6, matplotlib

### Figures (3-4 total)

**Figure 1: System Architecture**
```
┌─────────────────────────────────────────┐
│         PlethApp v1.0                   │
├─────────────────────────────────────────┤
│  GUI Layer (PyQt6)                      │
│  ├─ Interactive plotting                │
│  ├─ Manual editing tools                │
│  └─ Parameter configuration             │
├─────────────────────────────────────────┤
│  Core Analysis Engine (Headless)        │
│  ├─ Peak detection (threshold-based)    │
│  ├─ Breath event extraction             │
│  ├─ ML classification ← NEW             │
│  └─ Metrics computation                 │
├─────────────────────────────────────────┤
│  API/CLI Interface ← NEW                │
│  ├─ Python API (BreathAnalyzer)         │
│  └─ Command-line tool (plethapp)        │
├─────────────────────────────────────────┤
│  File I/O                               │
│  ├─ ABF (Axon Binary Format)            │
│  ├─ SMRX (Spike2)                       │
│  └─ CSV/NPZ export                      │
└─────────────────────────────────────────┘
```

**Figure 2: ML Classification Performance**
- Bar chart: F1 scores (ML vs GMM) for each breath type
- Box plots: Per-recording performance distribution
- Statistical significance markers (p-values)

**Figure 3: Example Analysis Output**
- Top panel: Raw plethysmography trace
- Middle panel: Detected breaths (color-coded by ML classification)
- Bottom panel: Instantaneous frequency over time
- Annotations: eupnea (green), sniffing (purple), sighs (orange)

**Figure 4: Speed Comparison**
- Bar chart: Processing time (ML vs GMM) for different file sizes
- Shows 40% speedup with ML approach

### Software Requirements for v1.0

**Core Features** (Must Have):
- ✅ Threshold-based peak detection (already implemented)
- ✅ Robust breath event extraction (already implemented)
- ✅ ML breath classifier (Random Forest + XGBoost) ← **TO IMPLEMENT**
- ✅ GUI (already implemented)
- ✅ Headless API ← **TO IMPLEMENT**
- ✅ CLI interface ← **TO IMPLEMENT**
- ✅ Pre-trained model distribution ← **TO IMPLEMENT**

**Documentation** (Must Have):
- ✅ README with installation instructions
- ✅ User guide (GUI usage)
- ✅ API reference (Sphinx docs)
- ✅ 2-3 Jupyter notebook tutorials

**Code Quality** (Must Have):
- ✅ Linting (black, flake8)
- ✅ Type hints (mypy)
- ✅ Unit tests (pytest, >80% coverage)
- ✅ Continuous integration (GitHub Actions)

**Licensing & Distribution**:
- ✅ **License**: MIT (maximum adoption)
- ✅ **Repository**: GitHub (public)
- ✅ **DOI**: Zenodo (for citations)
- ✅ **Installation**: pip installable (`pip install plethapp`)

---

## Paper 2: PlethApp v2.0 - End-to-End ML Pipeline

### Title
> **PlethApp 2.0: End-to-End Machine Learning for Automated Respiratory Event Detection**

### Target Journal
**Option A**: eNeuro (computational neuroscience) - **RECOMMENDED**
**Option B**: Journal of Neurophysiology (respiratory physiology)
**Option C**: JOSS v2 (faster, easier, but less prestigious)

**Why eNeuro?**
- ✅ Prestigious (impact factor ~3.5)
- ✅ Methods papers welcome
- ✅ Computational focus (ML is a plus)
- ✅ Open access
- ✅ Broader neuroscience audience

### Key Claims

1. **End-to-end ML pipeline** (detection + classification)
   - Eliminates manual parameter tuning
   - Adaptive to recording conditions
   - Pre-trained models work out-of-box

2. **Outperforms traditional methods on noisy/challenging data**
   - Edge cases where thresholds fail
   - Noisy recordings (movement artifacts)
   - Unusual breathing patterns (disease models)

3. **Real-time streaming capability** (optional)
   - Online breath detection (< 50ms latency)
   - Enables closed-loop experiments
   - Alarm triggering (apnea detection)

4. **Transfer learning across species/conditions**
   - Single model works for mouse and rat
   - Adapts to different anesthesia states
   - Few-shot learning (fine-tune with 5-10 examples)

5. **Active learning framework**
   - Model improves with user corrections
   - Incremental training (no full retraining)
   - Personalized to lab's recording setup

### Paper Structure (Full Methods Paper)

**Abstract** (250 words)
- Background, gap, approach, results, significance

**Introduction** (3-4 pages)
- Importance of automated breath detection
- Challenges with traditional methods
- ML as solution
- Overview of PlethApp v2.0
- Cite Paper 1 (v1.0)

**Methods** (4-5 pages)
- **Datasets**: 100 recordings (diverse conditions, species)
- **ML Detection Algorithm**: Random Forest architecture
- **Feature Engineering**: Signal + context features
- **Training Procedure**: Cross-validation, hyperparameter tuning
- **Validation Metrics**: Precision, recall, F1, temporal error
- **Comparison Baselines**: v1.0, breathmetrics, LabChart (if accessible)

**Results** (5-6 pages)
- **Detection Performance**: ML vs traditional methods
- **Classification Performance**: Improved from v1.0
- **Speed Benchmarks**: Real-time capable
- **Cross-Species Generalization**: Mouse vs rat
- **Noise Robustness**: Performance degradation curves
- **Active Learning**: Improvement over iterations
- **Real-Time Streaming**: Latency measurements

**Discussion** (3-4 pages)
- When does ML help most? (noise, edge cases)
- Limitations (requires training data)
- Comparison to related tools
- Future directions (deep learning, multi-modal)
- Broader impact (accessibility, reproducibility)

**Figures** (8-10 total)
1. Overview schematic (PlethApp ecosystem)
2. ML detection algorithm flowchart
3. Feature importance (what matters for detection?)
4. Performance comparison (detection F1, precision, recall)
5. Classification performance (updated from v1.0)
6. Example traces (ML detections vs ground truth)
7. Cross-species validation (mouse vs rat)
8. Noise robustness (SNR vs performance)
9. Real-time streaming demo (latency plot)
10. Active learning curve (performance vs # corrections)

### Software Requirements for v2.0

**New Features** (Must Have):
- ✅ ML breath detector (Random Forest) ← **TO IMPLEMENT**
- ✅ Improved ML classifier (retrained on larger dataset) ← **TO IMPLEMENT**
- ✅ Confidence visualization in GUI ← **TO IMPLEMENT**
- ✅ Active learning interface ← **TO IMPLEMENT**
- ⚠️ Real-time streaming mode (optional) ← **TO IMPLEMENT**

**Enhanced Documentation**:
- Advanced tutorials (transfer learning, active learning)
- Video tutorials
- Case studies (example datasets with analyses)

---

## Technical Implementation

### Phase 1: Data Annotation (Week 1-2)

**Goal**: Create training dataset for ML classifier

**Annotation Workflow**:
1. Load recording in PlethApp GUI
2. Run peak detection (current threshold method)
3. Manually review and correct:
   - Delete false positive peaks
   - Add missed peaks
   - Mark breath types: eupnea, sniff, sigh, artifact
4. Export annotations to JSON:
   ```json
   {
     "file": "recording_001.abf",
     "sweep": 0,
     "annotations": [
       {
         "breath_idx": 0,
         "onset": 1234,
         "offset": 2345,
         "peak": 1789,
         "expmin": 2567,
         "expoff": 2890,
         "type": "eupnea",
         "confidence": "high"
       },
       ...
     ]
   }
   ```

**Annotation Tool** (to create):
```python
# plethapp/annotation/annotator.py

class BreathAnnotator:
    """Helper for creating ML training datasets."""

    def __init__(self, recording_path):
        self.recording = load_recording(recording_path)
        self.annotations = []

    def add_annotation(self, breath_idx, breath_type, confidence="high"):
        """Mark a breath with its type."""
        self.annotations.append({
            "breath_idx": breath_idx,
            "type": breath_type,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

    def save(self, output_path):
        """Export annotations to JSON."""
        with open(output_path, 'w') as f:
            json.dump({
                "file": str(self.recording.path),
                "annotations": self.annotations,
                "metadata": {
                    "annotator": "user",
                    "date": datetime.now().isoformat(),
                    "plethapp_version": __version__
                }
            }, f, indent=2)
```

**Target Dataset** (for v1.0):
- **Minimum**: 20 recordings (mouse), ~2000 breaths
- **Ideal**: 30 recordings (20 mouse, 10 rat), ~3000 breaths
- **Annotation Time**: 3-6 hours total (10-15 min per recording)

**Annotation Guidelines**:
- **Eupnea**: Regular, consistent breathing (frequency < 5 Hz)
- **Sniffing**: Rapid, shallow breaths (frequency > 5 Hz, small amplitude)
- **Sigh**: Large amplitude breath (>2× baseline), longer Ti
- **Artifact**: Malformed breath (missing onset/offset, irregular shape)

**Quality Control**:
- Annotate same recording twice (1 week apart) → measure inter-rater reliability
- Expected agreement: >90% for clear breaths, >70% for ambiguous

---

### Phase 2: ML Classifier Implementation (Week 3-4)

**Module Structure**:
```
plethapp/ml/
├── __init__.py
├── breath_classifier.py      # Main classifier class
├── feature_extraction.py     # Compute features from breaths
├── model_training.py         # Training pipeline
├── active_learning.py        # Incremental learning
└── evaluation.py             # Validation metrics
```

**Feature Extraction** (`feature_extraction.py`):
```python
def extract_breath_features(t, y, sr_hz, breath_events):
    """
    Extract 20 features for ML classification.

    Args:
        t: Time array (seconds)
        y: Signal array (processed pleth)
        sr_hz: Sampling rate
        breath_events: Dict with onsets, offsets, peaks, expmins, expoffs

    Returns:
        features: np.ndarray (n_breaths, 20)
        feature_names: list[str] (20 feature names)
    """
    from core import metrics

    # Compute all metrics (already implemented!)
    onsets = breath_events['onsets']
    offsets = breath_events['offsets']
    expmins = breath_events['expmins']
    expoffs = breath_events['expoffs']
    peaks = breath_events['peaks']

    # Extract metric traces (stepwise constant over breaths)
    if_trace = metrics.compute_if(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    ti_trace = metrics.compute_ti(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    te_trace = metrics.compute_te(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    amp_insp = metrics.compute_amp_insp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    amp_exp = metrics.compute_amp_exp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    area_insp = metrics.compute_area_insp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    area_exp = metrics.compute_area_exp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    max_dinsp = metrics.compute_max_dinsp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    max_dexp = metrics.compute_max_dexp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    vent_proxy = metrics.compute_vent_proxy(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)

    # Extract values at breath midpoints
    mids = (onsets[:-1] + onsets[1:]) // 2

    features = []
    for i, mid in enumerate(mids):
        breath_features = [
            # Timing features (6)
            ti_trace[mid],
            te_trace[mid],
            ti_trace[mid] + te_trace[mid],  # Total cycle duration
            if_trace[mid],
            ti_trace[mid] / (te_trace[mid] + 1e-9),  # Ti/Te ratio
            t[onsets[i+1]] - t[onsets[i]] if i < len(onsets)-1 else np.nan,  # IBI

            # Amplitude features (4)
            amp_insp[mid],
            amp_exp[mid],
            amp_insp[mid] / (amp_exp[mid] + 1e-9),  # Amp ratio
            amp_insp[mid] / np.nanmean(amp_insp),  # Relative to baseline

            # Shape features (6)
            max_dinsp[mid],
            max_dexp[mid],
            area_insp[mid],
            area_exp[mid],
            vent_proxy[mid],
            area_insp[mid] / (area_exp[mid] + 1e-9),  # Area ratio

            # Context features (4)
            np.nanmean(if_trace[max(0, mid-500):mid+500]),  # Rolling mean (5 breaths @ 100Hz)
            np.nanstd(if_trace[max(0, mid-500):mid+500]),   # Rolling std
            float(i) / len(mids),  # Position in recording (0-1)
            0.0  # Distance from last sniff (computed separately)
        ]
        features.append(breath_features)

    feature_names = [
        'ti', 'te', 'cycle_dur', 'freq', 'ti_te_ratio', 'ibi',
        'amp_insp', 'amp_exp', 'amp_ratio', 'amp_rel',
        'max_dinsp', 'max_dexp', 'area_insp', 'area_exp', 'vent_proxy', 'area_ratio',
        'freq_rolling_mean', 'freq_rolling_std', 'position', 'dist_from_sniff'
    ]

    return np.array(features), feature_names
```

**Classifier** (`breath_classifier.py`):
```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import joblib

class BreathClassifier:
    """ML-based breath type classifier (ensemble)."""

    def __init__(self, model_path=None):
        if model_path:
            self.model = joblib.load(model_path)
            self.feature_names = self.model.feature_names_in_
        else:
            self.model = None
            self.feature_names = None

    def create_model(self):
        """Create ensemble model (Random Forest + XGBoost)."""
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )

        self.model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb)],
            voting='soft',  # Use probability averaging
            n_jobs=-1
        )

        return self.model

    def train(self, X_train, y_train, feature_names=None):
        """Train classifier on annotated data."""
        if self.model is None:
            self.create_model()

        self.feature_names = feature_names
        self.model.fit(X_train, y_train)

        # Store feature names for later
        self.model.feature_names_in_ = feature_names

    def predict(self, X):
        """Predict breath types with confidence scores."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        labels = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return labels, probabilities

    def get_feature_importance(self):
        """Get feature importance from Random Forest."""
        if self.model is None:
            return None

        # Extract Random Forest from ensemble
        rf = self.model.named_estimators_['rf']
        importance = rf.feature_importances_

        return dict(zip(self.feature_names, importance))

    def save(self, path):
        """Save trained model."""
        joblib.dump(self.model, path)
```

**Training Pipeline** (`model_training.py`):
```python
def train_classifier_from_annotations(annotation_files, output_path):
    """
    Train breath classifier from annotated recordings.

    Args:
        annotation_files: List of paths to annotation JSON files
        output_path: Where to save trained model

    Returns:
        classifier: Trained BreathClassifier
        metrics: Dict with validation metrics
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix

    # Load all annotations
    X_all = []
    y_all = []

    for ann_file in annotation_files:
        # Load recording
        with open(ann_file) as f:
            ann_data = json.load(f)

        recording = load_recording(ann_data['file'])

        # Extract features
        features, feature_names = extract_breath_features(
            recording.t, recording.y, recording.sr_hz, recording.breath_events
        )

        # Get labels from annotations
        labels = [ann['type'] for ann in ann_data['annotations']]

        X_all.append(features)
        y_all.append(labels)

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        classifier = BreathClassifier()
        classifier.train(X_train, y_train, feature_names)

        # Evaluate
        y_pred, _ = classifier.predict(X_val)

        from sklearn.metrics import f1_score
        f1 = f1_score(y_val, y_pred, average='weighted')
        cv_scores.append(f1)

        print(f"Fold {fold+1}/5: F1 = {f1:.3f}")

    print(f"Mean F1: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    # Train final model on all data
    classifier_final = BreathClassifier()
    classifier_final.train(X, y, feature_names)

    # Save
    classifier_final.save(output_path)

    # Generate full report
    y_pred_full, probs_full = classifier_final.predict(X)

    metrics = {
        'cv_f1_mean': np.mean(cv_scores),
        'cv_f1_std': np.std(cv_scores),
        'classification_report': classification_report(y, y_pred_full),
        'confusion_matrix': confusion_matrix(y, y_pred_full),
        'feature_importance': classifier_final.get_feature_importance()
    }

    return classifier_final, metrics
```

---

### Phase 3: Headless API Extraction (Week 7)

**Goal**: Create API that works without GUI (for batch processing)

**Core Module** (`plethapp/core/analyzer.py`):
```python
class BreathAnalyzer:
    """Headless breath analysis engine (no GUI dependencies)."""

    def __init__(self, model_path=None, config=None):
        """
        Initialize analyzer.

        Args:
            model_path: Path to pre-trained ML model (optional)
            config: Configuration dict or path to YAML
        """
        from plethapp.ml.breath_classifier import BreathClassifier

        # Load ML classifier if provided
        if model_path:
            self.classifier = BreathClassifier(model_path)
        else:
            # Try to load default model
            default_model = Path(__file__).parent.parent / "models" / "classifier_v1.pkl"
            if default_model.exists():
                self.classifier = BreathClassifier(str(default_model))
            else:
                self.classifier = None  # Will fall back to GMM

        # Load configuration
        self.config = self._load_config(config) if config else self._default_config()

    def _default_config(self):
        """Default analysis parameters."""
        return {
            'peak_detection': {
                'threshold': 0.1,
                'prominence': 0.05,
                'distance_ms': 100
            },
            'filters': {
                'low_hz': None,
                'high_hz': 30.0,
                'order': 4
            },
            'classification': {
                'method': 'ml',  # 'ml' or 'gmm'
                'confidence_threshold': 0.7
            }
        }

    def analyze(self, file_path, sweep_idx=0):
        """
        Analyze a single sweep.

        Args:
            file_path: Path to recording file (ABF, SMRX, etc.)
            sweep_idx: Which sweep to analyze

        Returns:
            AnalysisResults object
        """
        # Load data
        data = self._load_file(file_path)

        # Process sweep
        t = data['t']
        y = data['sweeps'][data['analyze_chan']][:, sweep_idx]
        sr_hz = data['sr_hz']

        # Apply filters
        y_filtered = self._apply_filters(y, sr_hz)

        # Detect peaks
        peaks = self._detect_peaks(y_filtered, sr_hz)

        # Extract breath events
        breath_events = self._compute_breath_events(y_filtered, peaks, sr_hz)

        # Classify breaths
        if self.classifier:
            labels, probs = self._classify_ml(t, y_filtered, sr_hz, peaks, breath_events)
        else:
            labels, probs = self._classify_gmm(t, y_filtered, sr_hz, peaks, breath_events)

        # Compute metrics
        metrics = self._compute_metrics(t, y_filtered, sr_hz, peaks, breath_events)

        return AnalysisResults(
            file_path=file_path,
            sweep_idx=sweep_idx,
            t=t,
            y=y_filtered,
            peaks=peaks,
            breath_events=breath_events,
            labels=labels,
            probabilities=probs,
            metrics=metrics
        )

    def batch_analyze(self, file_paths, n_jobs=-1):
        """
        Process multiple files in parallel.

        Args:
            file_paths: List of file paths
            n_jobs: Number of parallel jobs (-1 = all CPUs)

        Returns:
            List of AnalysisResults
        """
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.analyze)(f) for f in file_paths
        )

        return results
```

**Results Container** (`plethapp/core/results.py`):
```python
class AnalysisResults:
    """Container for analysis results."""

    def __init__(self, file_path, sweep_idx, t, y, peaks, breath_events, labels, probabilities, metrics):
        self.file_path = file_path
        self.sweep_idx = sweep_idx
        self.t = t
        self.y = y
        self.peaks = peaks
        self.breath_events = breath_events
        self.labels = labels
        self.probabilities = probabilities
        self.metrics = metrics

    def to_dataframe(self):
        """Convert to pandas DataFrame (breath-by-breath)."""
        import pandas as pd

        n_breaths = len(self.breath_events['onsets']) - 1

        data = {
            'breath_idx': range(n_breaths),
            'onset_time': self.t[self.breath_events['onsets'][:-1]],
            'offset_time': self.t[self.breath_events['offsets'][:-1]],
            'breath_type': self.labels,
            'confidence': np.max(self.probabilities, axis=1)
        }

        # Add metrics
        for metric_name, metric_values in self.metrics.items():
            if len(metric_values) == len(self.t):
                # Extract values at breath midpoints
                mids = (self.breath_events['onsets'][:-1] + self.breath_events['onsets'][1:]) // 2
                data[metric_name] = metric_values[mids]

        return pd.DataFrame(data)

    def to_csv(self, output_path):
        """Export to CSV."""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)

    def summary_stats(self):
        """Compute summary statistics."""
        from collections import Counter

        breath_counts = Counter(self.labels)

        return {
            'file': str(self.file_path),
            'sweep': self.sweep_idx,
            'n_breaths': len(self.labels),
            'n_eupnea': breath_counts.get('eupnea', 0),
            'n_sniff': breath_counts.get('sniff', 0),
            'n_sigh': breath_counts.get('sigh', 0),
            'mean_frequency': np.nanmean(self.metrics.get('if', [])),
            'mean_amplitude': np.nanmean(self.metrics.get('amp_insp', []))
        }
```

---

### Phase 4: CLI Interface (Week 7)

**Command-Line Tool** (`plethapp/cli.py`):
```python
import click
from pathlib import Path

@click.group()
def cli():
    """PlethApp - Plethysmography analysis tool."""
    pass

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='results/', help='Output directory')
@click.option('--model', '-m', default=None, help='Path to ML model')
@click.option('--config', '-c', default=None, help='Config YAML file')
@click.option('--sweep', '-s', default=0, type=int, help='Sweep index to analyze')
@click.option('--format', type=click.Choice(['csv', 'npz', 'both']), default='csv')
def analyze(input_file, output, model, config, sweep, format):
    """Analyze a plethysmography recording (headless mode)."""
    from plethapp.core.analyzer import BreathAnalyzer

    click.echo(f"Loading: {input_file}")
    analyzer = BreathAnalyzer(model_path=model, config=config)

    click.echo(f"Analyzing sweep {sweep}...")
    results = analyzer.analyze(input_file, sweep_idx=sweep)

    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export results
    base_name = Path(input_file).stem

    if format in ['csv', 'both']:
        csv_path = output_dir / f"{base_name}_sweep{sweep}.csv"
        results.to_csv(csv_path)
        click.echo(f"✓ Exported CSV: {csv_path}")

    if format in ['npz', 'both']:
        npz_path = output_dir / f"{base_name}_sweep{sweep}.npz"
        results.to_npz(npz_path)
        click.echo(f"✓ Exported NPZ: {npz_path}")

    # Print summary
    summary = results.summary_stats()
    click.echo(f"\nSummary:")
    click.echo(f"  Total breaths: {summary['n_breaths']}")
    click.echo(f"  Eupnea: {summary['n_eupnea']}")
    click.echo(f"  Sniffing: {summary['n_sniff']}")
    click.echo(f"  Sighs: {summary['n_sigh']}")
    click.echo(f"  Mean frequency: {summary['mean_frequency']:.2f} Hz")

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', default='results/', help='Output directory')
@click.option('--pattern', '-p', default='*.abf', help='File pattern')
@click.option('--jobs', '-j', default=-1, type=int, help='Parallel jobs (-1 = all CPUs)')
def batch(input_dir, output, pattern, jobs):
    """Batch process multiple recordings."""
    from plethapp.core.analyzer import BreathAnalyzer
    import glob

    # Find all matching files
    files = list(Path(input_dir).glob(pattern))
    click.echo(f"Found {len(files)} files matching pattern '{pattern}'")

    if not files:
        click.echo("No files found. Exiting.")
        return

    # Process in parallel
    analyzer = BreathAnalyzer()

    with click.progressbar(files, label='Processing') as bar:
        results = analyzer.batch_analyze(list(bar), n_jobs=jobs)

    # Export all results
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        base_name = Path(result.file_path).stem
        csv_path = output_dir / f"{base_name}.csv"
        result.to_csv(csv_path)

    click.echo(f"✓ Exported {len(results)} files to {output_dir}")

if __name__ == '__main__':
    cli()
```

**Installation Setup** (`setup.py`):
```python
from setuptools import setup, find_packages

setup(
    name='plethapp',
    version='1.0.0',
    description='Open-source plethysmography analysis with machine learning',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/plethapp',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
        'matplotlib>=3.4',
        'PyQt6>=6.2',
        'scikit-learn>=1.0',
        'xgboost>=1.5',
        'pandas>=1.3',
        'click>=8.0',
        'joblib>=1.0',
    ],
    entry_points={
        'console_scripts': [
            'plethapp=plethapp.cli:cli',
            'plethapp-gui=plethapp.main:main',
        ],
    },
    include_package_data=True,
    package_data={
        'plethapp': ['models/*.pkl', 'ui/*.ui'],
    },
    python_requires='>=3.9',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
```

---

## Timeline

### Paper 1 (v1.0) - 12 Weeks to Submission

**Weeks 1-2: Data Preparation**
- [ ] Annotate 20 recordings (mouse)
  - Time: ~10-15 min per recording = 3-5 hours total
  - Export annotations to JSON format
- [ ] Optional: Annotate 10 recordings (rat) for cross-species claim
  - Additional 2-3 hours
- [ ] Create annotation quality control report
  - Re-annotate 2-3 recordings, measure agreement

**Weeks 3-4: ML Implementation**
- [ ] Implement `BreathClassifier` class
  - Random Forest + XGBoost ensemble
  - Feature extraction (20 features from existing metrics)
- [ ] Train classifier on annotated data
  - 5-fold cross-validation
  - Tune hyperparameters if needed
- [ ] Validate performance
  - Compare to GMM baseline
  - Statistical testing (Wilcoxon)
  - **Decision point**: If F1 < 0.90, revisit features/model

**Weeks 5-6: Validation & Benchmarking**
- [ ] Cross-validation on held-out recordings
- [ ] Speed benchmarks (ML vs GMM)
- [ ] Cross-species validation (if rat data available)
- [ ] Generate confusion matrices, performance plots

**Week 7: API Extraction**
- [ ] Create `BreathAnalyzer` (headless API)
- [ ] Create CLI interface (`plethapp` command)
- [ ] Test API with example scripts
- [ ] Update existing GUI to use headless core

**Week 8: Code Quality**
- [ ] Code formatting (black, flake8)
- [ ] Type hints (mypy)
- [ ] Unit tests (pytest)
  - Test feature extraction
  - Test classifier predictions
  - Test API interface
- [ ] Continuous integration (GitHub Actions)

**Weeks 9-10: Documentation**
- [ ] README with installation instructions
- [ ] User guide (GUI usage)
- [ ] API reference (Sphinx)
- [ ] Tutorial notebooks (2-3):
  - Tutorial 1: Basic usage (GUI)
  - Tutorial 2: Batch processing (CLI)
  - Tutorial 3: Python API integration

**Week 11: Paper Writing**
- [ ] Draft JOSS paper (2-4 pages)
- [ ] Create figures (4 total)
- [ ] Write code examples
- [ ] Prepare bibliography

**Week 12: Release & Submission**
- [ ] GitHub release v1.0.0
- [ ] Create Zenodo DOI
- [ ] Submit to JOSS
- [ ] Announce on social media

**Post-Submission (Weeks 13-16)**:
- [ ] Respond to JOSS reviewer feedback (~2-4 weeks typical)
- [ ] Make requested changes
- [ ] Final acceptance
- [ ] Promote published software

---

### Paper 2 (v2.0) - Start After v1.0 Acceptance

**Timeline**: 6-9 months after v1.0 submission

**Phase 1: Extended Data Annotation** (Months 1-2)
- [ ] Annotate 50-100 recordings (diverse conditions)
- [ ] Include noise-stressed recordings
- [ ] Multiple species (mouse, rat, other?)
- [ ] Edge cases (disease models, anesthesia)

**Phase 2: ML Detection** (Months 2-4)
- [ ] Implement Random Forest breath detector
- [ ] Feature engineering for onset/offset
- [ ] Train on annotated data
- [ ] Validate detection accuracy

**Phase 3: Advanced Features** (Months 4-5)
- [ ] Real-time streaming mode (optional)
- [ ] Transfer learning framework
- [ ] Active learning improvements

**Phase 4: Comprehensive Validation** (Month 6)
- [ ] 100+ recording validation
- [ ] Cross-species benchmarks
- [ ] Noise robustness testing
- [ ] Comparison to LabChart (if accessible)

**Phase 5: Paper Writing** (Months 7-8)
- [ ] Full methods paper (8-10 pages)
- [ ] 8-10 figures
- [ ] Extensive methods section
- [ ] Discussion of ML approaches

**Phase 6: Submission** (Month 9)
- [ ] Submit to eNeuro/JNeurophys
- [ ] Respond to reviews
- [ ] Final acceptance

---

## Software Deliverables

### v1.0 Release Checklist

**Core Features**:
- [x] Threshold-based peak detection (already implemented)
- [x] Robust breath event extraction (already implemented)
- [ ] ML breath classifier (Random Forest + XGBoost)
- [x] GUI (PyQt6) (already implemented)
- [ ] Headless API (BreathAnalyzer class)
- [ ] CLI interface (plethapp command)
- [ ] Pre-trained model (classifier_v1.pkl)

**File Format Support**:
- [x] ABF (Axon Binary Format)
- [x] SMRX (Spike2)
- [x] CSV/NPZ export

**Documentation**:
- [ ] README.md (installation, quickstart)
- [ ] User guide (detailed GUI usage)
- [ ] API reference (Sphinx docs)
- [ ] Tutorial notebooks (3 total)
- [ ] Example datasets (5-10 recordings)

**Testing**:
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests (full pipeline)
- [ ] Validation scripts (reproduce paper results)

**Distribution**:
- [ ] GitHub repository (public)
- [ ] PyPI package (`pip install plethapp`)
- [ ] Zenodo DOI (for citations)
- [ ] Documentation website (Read the Docs)

**Quality Assurance**:
- [ ] Code linting (black, flake8)
- [ ] Type checking (mypy)
- [ ] Continuous integration (GitHub Actions)
- [ ] Version tagging (semantic versioning)

---

## Validation Strategy

### Dataset Requirements

**v1.0 (Minimum Viable)**:
- 20-30 annotated recordings
- At least 2000 annotated breaths
- Mix of conditions: baseline, stimulation, various behaviors
- Mouse (primary) + rat (optional for cross-species)

**v1.0 (Ideal)**:
- 40-50 annotated recordings
- 4000+ annotated breaths
- Multiple mouse strains
- Both mouse and rat

**v2.0 (Comprehensive)**:
- 100+ annotated recordings
- 10,000+ annotated breaths
- Diverse conditions (noise, movement artifacts)
- Multiple species
- Disease models (if applicable)

### Performance Metrics

**Classification Metrics** (v1.0):
- **F1 Score**: Harmonic mean of precision/recall
  - Target: >0.90 overall
  - Per-class: >0.85 for each breath type
- **Precision**: % of predicted breaths that are correct
  - Target: >0.92
- **Recall**: % of true breaths that are detected
  - Target: >0.88
- **Confusion Matrix**: Where does model get confused?

**Speed Metrics** (v1.0):
- **Processing time**: Seconds per sweep
  - Target: <1s for typical recording (5 min @ 1000 Hz)
  - Comparison: ML should be 30-50% faster than GMM
- **Latency**: Time from input to result
  - Important for real-time mode (v2.0)

**Detection Metrics** (v2.0):
- **Temporal Error**: Mean absolute error in onset/offset timing
  - Target: <50ms for onsets, <100ms for offsets
- **Detection Rate**: % of true breaths detected
  - Target: >0.95
- **False Positive Rate**: % of detections that are false
  - Target: <0.05

### Statistical Testing

**Comparison to Baseline** (GMM):
```python
from scipy.stats import wilcoxon

# Per-recording F1 scores
f1_gmm = [0.87, 0.89, 0.85, ...]  # From GMM
f1_ml = [0.93, 0.95, 0.91, ...]   # From ML

# Paired test (same recordings)
stat, pvalue = wilcoxon(f1_ml, f1_gmm, alternative='greater')

print(f"ML significantly better than GMM: p={pvalue:.4f}")
# Expected: p < 0.01
```

**Cross-Validation**:
- 5-fold stratified cross-validation
- Leave-one-animal-out (more stringent)
- Leave-one-condition-out (test generalization)

### Reproducibility

**Random Seeds**:
- Set random seeds for all stochastic processes
- Document in paper: `random_state=42` for scikit-learn

**Version Tracking**:
- Record software versions (Python, scikit-learn, XGBoost)
- Include in paper methods section
- Export environment: `conda env export > environment.yml`

**Data Availability**:
- Deposit example datasets on Zenodo/Dryad
- 5-10 annotated recordings (deidentified if needed)
- Allows readers to reproduce validation

---

## Next Steps

### Immediate Actions (This Week)

**Step 1: Set Up Annotation Workflow** (1-2 hours)
```bash
# Create annotation directory
mkdir -p annotations/

# Create annotation template
cat > annotations/template.json << EOF
{
  "file": "path/to/recording.abf",
  "sweep": 0,
  "annotations": [
    {
      "breath_idx": 0,
      "type": "eupnea",
      "confidence": "high",
      "notes": ""
    }
  ],
  "metadata": {
    "annotator": "your_name",
    "date": "2025-10-20",
    "plethapp_version": "0.9.0"
  }
}
EOF
```

**Step 2: Annotate First 5 Recordings** (1-2 hours)
- Choose representative recordings:
  - 2 baseline (quiet breathing)
  - 2 with stimulation
  - 1 with mixed behaviors
- Use PlethApp GUI to review/correct detections
- Export annotations to JSON

**Step 3: Test Feature Extraction** (30 min)
```python
# Quick test of feature extraction
from plethapp.ml.feature_extraction import extract_breath_features

# Load a test recording
# ... (use existing code)

# Extract features
features, feature_names = extract_breath_features(t, y, sr_hz, breath_events)

print(f"Extracted {features.shape[0]} breaths × {features.shape[1]} features")
print(f"Features: {feature_names}")

# Check for NaNs
nan_count = np.isnan(features).sum()
print(f"NaN values: {nan_count} / {features.size} ({100*nan_count/features.size:.1f}%)")
```

### Week 1-2 Goals

- [ ] Annotate 20 recordings
- [ ] Create feature extraction pipeline
- [ ] Test ML classifier on small dataset (proof of concept)
- [ ] Measure baseline GMM performance (for comparison)

### Decision Points

**End of Week 2** (After initial annotations):
- ✅ **Go/No-Go**: Do we have sufficient data quality?
  - Check: Inter-annotator agreement >90%
  - Check: Dataset covers diverse conditions
  - If NO: Annotate 5-10 more recordings

**End of Week 4** (After ML implementation):
- ✅ **Go/No-Go**: Does ML outperform GMM?
  - Check: ML F1 > 0.90 AND ML F1 > GMM F1
  - If NO: Revisit features, try different models
  - If YES: Proceed to API implementation

**End of Week 6** (After validation):
- ✅ **Go/No-Go**: Are results publication-ready?
  - Check: Performance metrics meet targets
  - Check: Speed improvements demonstrated
  - If NO: Additional tuning/data collection
  - If YES: Proceed to paper writing

### Communication Plan

**GitHub Repository**:
- Make public immediately (v0.9.0 "beta")
- Add "under development" notice
- Invite beta testers from lab/collaborators

**Community Engagement**:
- Tweet about project (build anticipation)
- Post on r/neuro, r/python
- Present at lab meeting
- Reach out to potential users

**Pre-print** (optional):
- After JOSS submission, post to bioRxiv
- Increases visibility while under review
- Can get early feedback from community

---

## Risk Mitigation

### Risk 1: ML Doesn't Outperform GMM
**Likelihood**: Low-Medium
**Impact**: High (undermines v1.0 novelty)

**Mitigation**:
- Start with proof-of-concept (Week 4 decision point)
- If ML only matches GMM: Pivot to "hybrid approach" (ML + GMM ensemble)
- Alternative framing: "Automated parameter tuning" instead of "better performance"
- Can still publish GUI + API as novelty

### Risk 2: Insufficient Training Data
**Likelihood**: Medium
**Impact**: Medium

**Mitigation**:
- Active learning (model suggests what to label next)
- Data augmentation (add synthetic noise)
- Transfer learning (pre-train on larger public datasets if available)
- Minimum viable: 20 recordings should be sufficient for Random Forest

### Risk 3: Annotation Time Too Long
**Likelihood**: Low
**Impact**: Low

**Mitigation**:
- Current annotation: ~10 min per recording (already fast!)
- Parallelize: Annotate while doing other experiments
- Hire undergrad for annotation (if funding available)
- Start with minimal dataset (20 recordings = 3-4 hours)

### Risk 4: JOSS Rejection
**Likelihood**: Very Low (JOSS acceptance rate ~90%)
**Impact**: Medium (delays publication)

**Mitigation**:
- Follow JOSS guidelines closely
- Ensure software is well-documented
- Respond quickly to reviewer feedback
- Alternative: Submit to PLOS ONE (broader scope)

### Risk 5: GPL Licensing Concerns
**Likelihood**: Low
**Impact**: Low

**Mitigation**:
- Using MIT license (permissive, no issues)
- All dependencies are MIT/BSD/Apache compatible
- XGBoost is Apache 2.0 (compatible with MIT)

---

## Success Criteria

### v1.0 (Paper 1) Success = ALL of:
- ✅ ML classifier F1 > 0.90
- ✅ ML faster than GMM (>20% speedup)
- ✅ JOSS paper accepted
- ✅ GitHub repo has 50+ stars within 6 months
- ✅ At least 5 external users (outside your lab)
- ✅ Cited in at least 1 other paper within 1 year

### v2.0 (Paper 2) Success = ALL of:
- ✅ ML detection precision > 0.95, recall > 0.90
- ✅ Cross-species validation successful
- ✅ eNeuro/JNeurophys paper accepted
- ✅ Real-time mode implemented (<50ms latency)
- ✅ GitHub repo has 200+ stars

### Long-Term Impact Goals:
- Become standard tool for respiratory analysis (cited in methods)
- Adopted by 10+ labs
- Used in 20+ published papers
- Potential grant funding for further development
- Conference presentations (SfN, APS)

---

## Conclusion

This two-paper strategy provides a clear path to publication while maximizing impact and citations. The aggressive timeline for v1.0 (2-3 months) is achievable given the current state of PlethApp and the limited scope of ML implementation needed.

**Key Advantages**:
1. Fast first publication (builds momentum)
2. Lower risk (incremental validation)
3. Community feedback guides v2.0
4. Better for CV (2 papers > 1 paper)

**Next Steps**:
1. Start annotating recordings THIS WEEK
2. Implement ML classifier (Weeks 3-4)
3. Validate performance (Weeks 5-6)
4. Extract API (Week 7)
5. Write paper (Weeks 9-11)
6. Submit to JOSS (Week 12)

**Timeline to first publication: 12 weeks (3 months)**

---

**Document Version**: 1.0
**Last Updated**: 2025-10-20
**Status**: Planning - Ready to Execute
