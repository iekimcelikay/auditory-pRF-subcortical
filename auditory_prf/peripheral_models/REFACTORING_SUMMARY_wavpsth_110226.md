# Cochlea Simulation Module - Refactoring Summary

**Date:** February 2026  
**Status:** ✅ Complete

## Table of Contents
1. [Overview](#overview)
2. [Folder Structure](#folder-structure)
3. [Module Descriptions](#module-descriptions)
4. [Classes and Methods](#classes-and-methods)
5. [Function Reference](#function-reference)
6. [Usage Examples](#usage-examples)
7. [Backward Compatibility](#backward-compatibility)
8. [Migration Guide](#migration-guide)

---

## Overview

The `cochlea_simulation.py` module has been refactored from a monolithic 702-line file into a modular architecture with focused responsibilities. This refactoring improves:

- **Maintainability**: Each module has a single, clear purpose
- **Testability**: Smaller, isolated units are easier to test
- **Readability**: Reduced cognitive load with focused modules
- **Extensibility**: Easier to add new simulation types

### Summary Statistics

| Metric | Before | After |
|--------|--------|-------|
| **Total Files** | 1 | 5 |
| **Lines of Code** | 702 | 775 (split across files) |
| **Classes** | 4 | 4 (same classes, reorganized) |
| **Main Module Size** | 702 lines | 77 lines (compatibility layer) |

---

## Folder Structure

```
peripheral_models/
│
├── cochlea_config.py              # Configuration dataclass for cochlea model
├── tone_config.py                 # Configuration dataclass for tone stimuli
├── cochlea_processor.py           # Core processing logic for Zilany2014 model
│
├── simulation_base.py             # ⭐ NEW: Base class with shared logic
│   └── _SimulationBase
│
├── tone_simulation.py             # ⭐ NEW: Tone generation simulation
│   └── CochleaSimulation
│
├── wav_simulation_mean.py         # ⭐ NEW: WAV mean rate processing
│   └── CochleaWavSimulationMean
│
├── wav_simulation_psth.py         # ⭐ NEW: WAV PSTH processing
│   └── CochleaWavSimulation
│
└── cochlea_simulation.py          # ⭐ UPDATED: Compatibility re-export module
    └── (re-exports all classes)
```

### File Sizes

| File | Lines | Purpose |
|------|-------|---------|
| `simulation_base.py` | 93 | Base class with shared functionality |
| `tone_simulation.py` | 192 | Tone generation simulation |
| `wav_simulation_mean.py` | 201 | WAV file processing with analytical mean rates |
| `wav_simulation_psth.py` | 212 | WAV file processing with full PSTH calculation |
| `cochlea_simulation.py` | 77 | Backward compatibility layer |

---

## Module Descriptions

### 1. `simulation_base.py`

**Purpose:** Base class containing shared logic for all simulation types.

**Key Features:**
- Logging configuration and setup
- Metadata saving utilities
- Result file saving in multiple formats (NPZ, PKL, MAT)
- Runtime information tracking
- Common initialization logic

**Dependencies:**
- `peripheral_models.cochlea_config`
- `peripheral_models.cochlea_processor`
- `utils.logging_configurator`
- `utils.metadata_saver`
- `utils.result_saver`

---

### 2. `tone_simulation.py`

**Purpose:** Simulation for programmatically generated tone stimuli.

**Key Features:**
- Tone generation using `SoundGen`
- Custom folder naming with simulation parameters
- Integration with tone parameter sweeps
- Efficient processing with memory management

**Dependencies:**
- `peripheral_models.simulation_base`
- `peripheral_models.cochlea_config`
- `peripheral_models.tone_config`
- `stimuli.soundgen`
- `utils.folder_management`
- `utils.stimulus_utils`

**Typical Use Case:**
- Frequency response characterization
- dB level sensitivity analysis
- Harmonic complex tone testing

---

### 3. `wav_simulation_mean.py`

**Purpose:** Process WAV files using analytical mean rate calculation.

**Key Features:**
- Uses `run_zilany2014_rate` for efficient computation
- Automatic WAV file resampling
- Filename parsing for metadata extraction
- Population rate calculation
- Memory-efficient streaming processing

**Dependencies:**
- `peripheral_models.simulation_base`
- `peripheral_models.cochlea_config`
- `soundfile` for WAV I/O
- `thorns.waves` for resampling
- `utils.calculate_population_rate`
- `utils.timestamp_utils`

**Typical Use Case:**
- Natural sound processing
- Speech or animal vocalization analysis
- Fast mean rate estimation (no spike trains)

---

### 4. `wav_simulation_psth.py`

**Purpose:** Process WAV files with full PSTH (Peri-Stimulus Time Histogram) calculation.

**Key Features:**
- Uses `run_zilany2014` for complete spike train modeling
- Generates both PSTH and mean rates
- Automatic WAV file resampling
- Filename parsing for metadata extraction
- Population rate from both PSTH and mean rates

**Dependencies:**
- `peripheral_models.simulation_base`
- `peripheral_models.cochlea_config`
- `soundfile` for WAV I/O
- `thorns.waves` for resampling
- `utils.calculate_population_rate`
- `utils.timestamp_utils`

**Typical Use Case:**
- Detailed temporal response analysis
- Spike train statistics
- PSTH visualization and analysis
- Phase-locking studies

---

### 5. `cochlea_simulation.py` (Compatibility Module)

**Purpose:** Maintains backward compatibility by re-exporting all classes.

**Key Features:**
- Re-exports all simulation classes
- Preserves original `__main__` test block
- `__all__` export list for `from module import *`
- Zero breaking changes for existing code

**Exports:**
- `_SimulationBase`
- `CochleaSimulation`
- `CochleaWavSimulationMean`
- `CochleaWavSimulation`

---

## Classes and Methods

### `_SimulationBase` (Base Class)

**File:** `simulation_base.py`  
**Purpose:** Shared logic for all simulation types. Not meant to be used directly.

#### Methods

| Method | Type | Description | Usage |
|--------|------|-------------|-------|
| `__init__(config)` | Constructor | Initialize with CochleaConfig | Called by subclass constructors |
| `_setup_logging_and_savers()` | Private | Configure logging and file savers | Called by subclasses during setup |
| `_save_metadata(data, base_filename)` | Private | Save metadata in JSON/YAML/TXT format | Called to save configuration and runtime info |
| `_save_single_result(data, filename_base)` | Private | Save results in NPZ/PKL/MAT formats | Called after each stimulus is processed |
| `_save_runtime_info(elapsed_time, stim_count)` | Private | Save timing and performance metrics | Called at end of simulation run |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | `CochleaConfig` | Model configuration parameters |
| `processor` | `CochleaProcessor` | Cochlea model processor instance |
| `results` | `dict` | Lightweight result references |
| `save_dir` | `Path` | Output directory path |
| `metadata_saver` | `MetadataSaver` | Metadata file handler |
| `result_saver` | `ResultSaver` | Result file handler |

---

### `CochleaSimulation` (Tone Simulation)

**File:** `tone_simulation.py`  
**Inherits:** `_SimulationBase`  
**Purpose:** Generate and process tone stimuli through the cochlea model.

#### Methods

| Method | Type | Description | Usage |
|--------|------|-------------|-------|
| `__init__(config, tone_config)` | Constructor | Initialize with model and tone configs | Create simulation instance |
| `_custom_folder_name_builder(params, timestamp)` | Private | Build descriptive folder name | Called by folder management |
| `setup_output_folder()` | Public | Create output directory and setup logging | Called before running simulation |
| `run()` | Public | **Main entry point** - Execute full simulation pipeline | Run the simulation |

#### Constructor Parameters

```python
CochleaSimulation(
    config: CochleaConfig,      # Model parameters
    tone_config: ToneConfig     # Stimulus parameters
)
```

#### Workflow

1. Initialize with configurations
2. Call `setup_output_folder()` (optional, auto-called by `run()`)
3. Call `run()` to process all tone combinations
4. Results saved automatically to disk
5. Returns dictionary of result references

---

### `CochleaWavSimulationMean` (WAV Mean Rate)

**File:** `wav_simulation_mean.py`  
**Inherits:** `_SimulationBase`  
**Purpose:** Process WAV files using efficient analytical mean rate calculation.

#### Methods

| Method | Type | Description | Usage |
|--------|------|-------------|-------|
| `parse_wav_filename(filename)` | Static | Parse metadata from WAV filename | Extract sound info from filename |
| `__init__(config, wav_files, ...)` | Constructor | Initialize with WAV file list | Create simulation instance |
| `setup_output_folder()` | Public | Create output directory and setup logging | Called before running simulation |
| `run()` | Public | **Main entry point** - Process all WAV files | Run the simulation |

#### Constructor Parameters

```python
CochleaWavSimulationMean(
    config: CochleaConfig,              # Model parameters
    wav_files: list[Path],              # List of WAV file paths
    metadata_dict: dict = None,         # Optional: explicit metadata
    auto_parse: bool = False,           # Auto-parse filenames
    parser_func: callable = None        # Custom parser function
)
```

#### Workflow

1. Initialize with configuration and WAV file list
2. Optionally parse filenames for metadata
3. Call `run()` to process all files
4. Each file: load → resample → process → calculate population rate → save
5. Returns dictionary of result references

---

### `CochleaWavSimulation` (WAV PSTH)

**File:** `wav_simulation_psth.py`  
**Inherits:** `_SimulationBase`  
**Purpose:** Process WAV files with full PSTH calculation using spike train model.

#### Methods

| Method | Type | Description | Usage |
|--------|------|-------------|-------|
| `parse_wav_filename(filename)` | Static | Parse metadata from WAV filename | Extract sound info from filename |
| `__init__(config, wav_files, ...)` | Constructor | Initialize with WAV file list | Create simulation instance |
| `setup_output_folder()` | Public | Create output directory and setup logging | Called before running simulation |
| `run()` | Public | **Main entry point** - Process all WAV files | Run the simulation |

#### Constructor Parameters

```python
CochleaWavSimulation(
    config: CochleaConfig,              # Model parameters
    wav_files: list[Path],              # List of WAV file paths
    metadata_dict: dict = None,         # Optional: explicit metadata
    auto_parse: bool = False,           # Auto-parse filenames
    parser_func: callable = None        # Custom parser function
)
```

#### Workflow

1. Initialize with configuration and WAV file list
2. Optionally parse filenames for metadata
3. Call `run()` to process all files
4. Each file: load → resample → process → generate PSTH → calculate rates → save
5. Returns dictionary of result references

---

## Function Reference

### Base Class Functions (`_SimulationBase`)

#### `_setup_logging_and_savers()`

**Purpose:** Configure logging and initialize file saver instances.

**Details:**
- Sets up file and console logging with configurable levels
- Creates `MetadataSaver` for config files
- Creates `ResultSaver` for data files
- Uses configuration from `CochleaConfig`

**Called By:** Subclass `setup_output_folder()` methods

---

#### `_save_metadata(data: dict, base_filename: str)`

**Purpose:** Save metadata in configured format (JSON/YAML/TXT).

**Parameters:**
- `data`: Dictionary of metadata to save
- `base_filename`: Filename without extension

**Output Formats:**
- JSON: Structured, machine-readable
- YAML: Human-readable, with comments
- TXT: Simple key-value pairs

**Called By:**
- `setup_output_folder()` → saves simulation config
- `_save_runtime_info()` → saves timing data

---

#### `_save_single_result(data: dict, filename_base: str)`

**Purpose:** Save simulation results in one or more formats.

**Parameters:**
- `data`: Dictionary containing arrays and metadata
- `filename_base`: Filename without extension

**Output Formats:**
- NPZ: NumPy compressed, fast I/O
- PKL: Python pickle, general purpose
- MAT: MATLAB format for cross-platform use

**Typical Data Structure:**
```python
{
    'freq': float,              # Stimulus frequency (tone)
    'db': float,                # Stimulus level
    'cf_list': ndarray,         # Characteristic frequencies
    'duration': float,          # Stimulus duration
    'time_axis': ndarray,       # Time points (PSTH only)
    'mean_rates': dict,         # Mean rates by fiber type
    'psth': dict,               # PSTH data by fiber type
    'soundfileid': str,         # Identifier (WAV only)
    'metadata': dict            # Additional metadata
}
```

---

#### `_save_runtime_info(elapsed_time: float, stim_count: int)`

**Purpose:** Save performance metrics after simulation completes.

**Parameters:**
- `elapsed_time`: Total simulation time in seconds
- `stim_count`: Number of stimuli processed

**Output:**
- Formatted time string (e.g., "2h 34m 12.50s")
- Number of stimuli
- Total results count

---

### Tone Simulation Functions (`CochleaSimulation`)

#### `_custom_folder_name_builder(params: dict, timestamp: str) -> str`

**Purpose:** Build descriptive folder name with simulation parameters.

**Parameters:**
- `params`: Dictionary with `num_cf`, `num_ANF`, `num_tones`
- `timestamp`: Formatted timestamp string

**Returns:** 
- Folder name like: `results_40cf_128-128-128anf_10tones_20260211_182200`

**Format:**
- `{num_cf}cf`: Number of characteristic frequencies
- `{lsr}-{msr}-{hsr}anf`: ANF fiber counts (low/mid/high SR)
- `{num_tones}tones`: Number of tone stimuli
- `{timestamp}`: Date and time

---

#### `setup_output_folder()`

**Purpose:** Create organized output directory with metadata.

**Actions:**
1. Use `FolderManager` to create timestamped folder
2. Setup logging and savers
3. Save simulation configuration metadata
4. Log output directory path

**Configuration Saved:**
- Model parameters (fs, CF range, ANF counts, etc.)
- Stimulus parameters (sample rate, duration, harmonics, etc.)
- Processing type identifier

---

#### `run() -> dict`

**Purpose:** Main simulation pipeline - generate and process all tone stimuli.

**Workflow:**
1. Setup output folder (if not already done)
2. Generate tone stimuli using `tone_generator`
3. Process each stimulus through Zilany2014 model
4. Save results immediately after each stimulus
5. Clear large arrays from memory
6. Save runtime info at completion

**Returns:**
- Dictionary with lightweight result references (no large arrays)
- Keys: `stim_id` (e.g., "freq_125.0hz_db_60")
- Values: `{'freq', 'db', 'saved_to', 'cf_list'}`

**Memory Management:**
- Results saved immediately to disk
- Large arrays deleted after saving
- Garbage collection after each stimulus

---

### WAV Mean Rate Functions (`CochleaWavSimulationMean`)

#### `parse_wav_filename(filename: str) -> dict` [static]

**Purpose:** Extract metadata from WAV filename following naming convention.

**Expected Format:** `s3_animal_1_ramp10.wav`

**Returns:**
```python
{
    'sound_number': 's3',
    'sound_type': 'animal',
    'type_number': '1',
    'ramp_id': 'ramp10'
}
```

**Fallback:** Returns empty dict if parsing fails (with warning)

---

#### `setup_output_folder()`

**Purpose:** Create output directory for WAV mean rate processing.

**Folder Name Format:**
- `wav_{num_cf}cf_analyticalmeanrate_{num_files}files_{timestamp}`

**Actions:**
1. Create timestamped directory
2. Setup logging and savers
3. Save configuration metadata
4. Log output path

---

#### `run() -> dict`

**Purpose:** Process all WAV files using analytical mean rate model.

**Workflow:**
1. Setup output folder (if not done)
2. For each WAV file:
   - Load audio using `soundfile`
   - Resample if needed using `thorns.waves`
   - Process using `processor.process_wav_meanrate()`
   - Calculate population rate
   - Save results immediately
   - Clear memory
3. Save runtime info
4. Return result references

**Model Used:** `run_zilany2014_rate` (analytical, no spike trains)

**Advantages:**
- Fast computation (no spike train generation)
- Lower memory usage
- Suitable for mean rate analysis only

---

### WAV PSTH Functions (`CochleaWavSimulation`)

#### `parse_wav_filename(filename: str) -> dict` [static]

**Purpose:** Extract metadata from WAV filename (same as mean rate version).

**Expected Format:** `s3_animal_1_ramp10.wav`

**Returns:**
```python
{
    'sound_number': 's3',
    'sound_type': 'animal',
    'type_number': '1',
    'ramp_id': 'ramp10'
}
```

---

#### `setup_output_folder()`

**Purpose:** Create output directory for WAV PSTH processing.

**Folder Name Format:**
- `wav_{num_cf}cf_{lsr}-{msr}-{hsr}anf_psth_{num_files}files_{timestamp}`

**Actions:**
1. Create timestamped directory
2. Setup logging and savers
3. Save configuration metadata (includes ANF counts)
4. Log output path

---

#### `run() -> dict`

**Purpose:** Process all WAV files with full PSTH calculation.

**Workflow:**
1. Setup output folder (if not done)
2. For each WAV file:
   - Load audio using `soundfile`
   - Resample if needed using `thorns.waves`
   - Process using `processor.process_wav_psth()`
   - Calculate population rate from PSTH
   - Calculate population rate from mean rates
   - Save results immediately
   - Clear memory
3. Save runtime info
4. Return result references

**Model Used:** `run_zilany2014` (full spike train model)

**Advantages:**
- Complete temporal information
- PSTH for visualization
- Spike train statistics
- Both PSTH and mean rates available

**Trade-offs:**
- Slower than mean rate only
- Higher memory usage
- Larger output files

---

## Usage Examples

### Example 1: Tone Generation Simulation

```python
from peripheral_models.cochlea_config import CochleaConfig
from peripheral_models.tone_config import ToneConfig
from peripheral_models.cochlea_simulation import CochleaSimulation

# Configure cochlea model
model_config = CochleaConfig(
    num_cf=40,
    num_ANF=(128, 128, 128),  # LSR, MSR, HSR counts
    powerlaw='approximate',
    output_dir="./output/tone_simulation",
    experiment_name="frequency_sweep",
    save_formats=['npz'],
    save_mean_rates=True,
    save_psth=False,
    log_console_level='INFO'
)

# Configure tone stimuli
tone_config = ToneConfig(
    db_range=[40, 60, 80],              # Three dB levels
    freq_range=(125, 8000, 20),         # 20 frequencies from 125-8000 Hz
    tone_duration=0.200,                # 200 ms
    num_harmonics=1,                    # Pure tones
    sample_rate=100000,
    tau_ramp=0.005
)

# Run simulation
simulation = CochleaSimulation(model_config, tone_config)
results = simulation.run()

# Results saved to disk automatically
print(f"Processed {len(results)} stimuli")
print(f"Output: {simulation.save_dir}")
```

---

### Example 2: WAV File Mean Rate Processing

```python
from pathlib import Path
from peripheral_models.cochlea_config import CochleaConfig
from peripheral_models.cochlea_simulation import CochleaWavSimulationMean

# Configure model
config = CochleaConfig(
    num_cf=40,
    num_ANF=(128, 128, 128),
    peripheral_fs=100000,
    output_dir="./output/wav_meanrate",
    experiment_name="speech_analysis",
    save_formats=['npz', 'mat'],
    save_mean_rates=True,
    save_psth=False
)

# Get WAV files
wav_dir = Path("./stimuli/speech")
wav_files = sorted(wav_dir.glob("*.wav"))

# Run simulation with automatic filename parsing
simulation = CochleaWavSimulationMean(
    config, 
    wav_files, 
    auto_parse=True  # Parse metadata from filenames
)
results = simulation.run()

print(f"Processed {len(results)} WAV files")
```

---

### Example 3: WAV File PSTH Processing

```python
from pathlib import Path
from peripheral_models.cochlea_config import CochleaConfig
from peripheral_models.cochlea_simulation import CochleaWavSimulation

# Configure model for PSTH
config = CochleaConfig(
    num_cf=40,
    num_ANF=(128, 128, 128),
    peripheral_fs=100000,
    output_dir="./output/wav_psth",
    experiment_name="vocalization_psth",
    save_formats=['npz'],
    save_mean_rates=True,
    save_psth=True,  # Enable PSTH saving
    fs_target=1000   # Downsample PSTH to 1 kHz
)

# Get WAV files
wav_files = [Path("./stimuli/animal_call_1.wav")]

# Provide explicit metadata
metadata = {
    'animal_call_1': {
        'species': 'bird',
        'call_type': 'song',
        'recording_date': '2026-01-15'
    }
}

# Run simulation
simulation = CochleaWavSimulation(
    config,
    wav_files,
    metadata_dict=metadata
)
results = simulation.run()

print(f"PSTH data saved to {simulation.save_dir}")
```

---

### Example 4: Custom WAV Filename Parser

```python
from peripheral_models.cochlea_simulation import CochleaWavSimulationMean

def custom_parser(filename: str) -> dict:
    """Parse custom filename format: exp12_subj03_trial05.wav"""
    parts = filename.replace('.wav', '').split('_')
    return {
        'experiment': parts[0],
        'subject': parts[1],
        'trial': parts[2]
    }

# Use custom parser
simulation = CochleaWavSimulationMean(
    config,
    wav_files,
    auto_parse=True,
    parser_func=custom_parser
)
results = simulation.run()
```

---

## Backward Compatibility

### No Changes Required

All existing code continues to work without modification:

```python
# ✅ Old import style - STILL WORKS
from peripheral_models.cochlea_simulation import (
    CochleaSimulation,
    CochleaWavSimulation,
    CochleaWavSimulationMean,
    _SimulationBase
)
```

### New Import Options

You can now import from specific modules for clarity:

```python
# ✅ New import style - MORE EXPLICIT
from peripheral_models.tone_simulation import CochleaSimulation
from peripheral_models.wav_simulation_psth import CochleaWavSimulation
from peripheral_models.wav_simulation_mean import CochleaWavSimulationMean
from peripheral_models.simulation_base import _SimulationBase
```

### `from module import *`

The `__all__` list is properly configured:

```python
# ✅ Works correctly
from peripheral_models.cochlea_simulation import *
# Imports: _SimulationBase, CochleaSimulation, 
#          CochleaWavSimulationMean, CochleaWavSimulation
```

---

## Migration Guide

### For Existing Code

**No migration needed!** Your code will work exactly as before.

### For New Code

**Recommended:** Use explicit imports from specific modules:

**Before:**
```python
from peripheral_models.cochlea_simulation import CochleaSimulation
```

**After (recommended):**
```python
from peripheral_models.tone_simulation import CochleaSimulation
```

**Benefits:**
- More explicit dependencies
- Easier to understand module structure
- Better IDE support and autocomplete
- Clearer which functionality you're using

### For Library Maintainers

When adding new simulation types:

1. Create a new file in `peripheral_models/`
2. Inherit from `_SimulationBase`
3. Implement required methods:
   - `__init__()`
   - `setup_output_folder()`
   - `run()`
4. Add import to `cochlea_simulation.py` for compatibility
5. Add to `__all__` list

---

## Testing

### Verification Checklist

- ✅ All Python files compile without syntax errors
- ✅ All classes properly inherit from `_SimulationBase`
- ✅ All expected methods present in each class
- ✅ Imports work from both old and new locations
- ✅ Example files compile successfully
- ✅ `__main__` test block preserved and functional
- ✅ `__all__` export list correct
- ✅ Backward compatibility maintained

### Running Tests

```bash
# Test imports
python -c "from peripheral_models.cochlea_simulation import *; print('✓ Imports OK')"

# Run example
python peripheral_models/cochlea_simulation.py

# Test tone simulation
python examples/cochlea_soundgen_example.py

# Test WAV simulation
python examples/cochlea_wav_example.py
```

---

## Benefits of Refactoring

### 1. Improved Organization
- **Before:** 702 lines in one file
- **After:** 5 focused modules (~77-212 lines each)

### 2. Better Maintainability
- Each module has single responsibility
- Easier to locate and fix bugs
- Changes isolated to relevant module

### 3. Enhanced Testability
- Smaller units easier to test
- Can test base class separately
- Can mock dependencies more easily

### 4. Clearer Dependencies
- Explicit imports show what each module needs
- Easier to understand module relationships
- Better for dependency management

### 5. Reduced Cognitive Load
- Focus on one simulation type at a time
- Don't need to read entire monolithic file
- Clear inheritance hierarchy

### 6. Extensibility
- Easy to add new simulation types
- Follow existing pattern
- Reuse base class functionality

---

## Summary

The refactoring successfully split the monolithic `cochlea_simulation.py` into a modular architecture while maintaining complete backward compatibility. The new structure provides:

- **4 focused modules** for different simulation types
- **1 base class** with shared functionality  
- **1 compatibility layer** for existing code
- **Zero breaking changes**
- **Improved maintainability and extensibility**

All existing code continues to work unchanged, while new code can benefit from the clearer module structure and explicit imports.

---

**For questions or issues, refer to:**
- Module source code with inline documentation
- Example files in `examples/` directory
- Git history for detailed change information

**Last Updated:** February 11, 2026
