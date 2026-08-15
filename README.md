# Near-Field 3-D Bird Sound Localization

`localize.py` localizes simulated bird vocalizations recorded by a planar,
multichannel microphone array. It evaluates a Cartesian grid of candidate source
positions with three wideband localization methods:

- **SRP-PHAT** (Steered Response Power with Phase Transform)
- **DAS/Bartlett** (Delay-and-Sum beamforming)
- **MVDR/Capon** (Minimum Variance Distortionless Response)

The program supports recordings containing either one or three sources. The
source count is known metadata and must be supplied with `--num_sources`; it is
never estimated from the spatial score map.

## Features

- near-field, three-dimensional Cartesian search;
- deterministic microphone selection by farthest-point sampling;
- configurable STFT, frequency band, activity filtering, grid, and batch size;
- 26-connected regional-maximum extraction with deterministic plateau handling;
- exact fixed-count peak selection for three-source recordings;
- optional truth matching and error calculation using the Hungarian algorithm;
- normalized orthogonal score-map projections and relevant horizontal slices;
- resumable-by-recording tabular output through an incrementally written CSV.

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- SciPy
- SoundFile
- Matplotlib

Create and activate a virtual environment in PowerShell, then install the
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scipy soundfile matplotlib
```

On Linux or macOS, activate the environment with
`source .venv/bin/activate` instead.

## Input files

The input data are not bundled with this repository. A run requires a microphone
coordinate file and either one WAV file or a glob matching multiple WAV files.

### Microphone coordinates

The CSV passed to `--csv` must contain the columns `mic_id`, `x`, `y`, and `z`:

```csv
mic_id,x,y,z
centre,1.24,2.10,1.60
001,0.0,-0.50,-0.40
002,0.0,-0.25,-0.40
```

- Exactly one `centre` row gives the array centre in absolute room coordinates.
- Numeric microphone rows give coordinates relative to that centre.
- Numeric IDs are sorted numerically, so audio channel 0 corresponds to the
  lowest numeric microphone ID, channel 1 to the next ID, and so on.
- The WAV channel count must equal the number of numeric microphone rows.

Coordinates are measured in metres and use the order `(x, y, z)`. For the
project's planar array, `x` is perpendicular to the array and `y` and `z` lie in
the microphone plane.

### Audio

WAV files must contain one channel for every numeric microphone row. SoundFile
loads audio as a two-dimensional array with shape `(samples, channels)`. The
selected audio channels and selected microphone coordinates use the same index
array, preserving their alignment.

### Optional ground-truth metadata

If a matching text file is available, it may contain one or three source
coordinates in lines containing `[x, y, z]`, for example:

```text
source_1_position = [3.40, 1.70, 1.20]
```

For a single WAV, provide the metadata path with `--txt`. Otherwise the program
looks for a `.txt` file with the same basename as the WAV. A missing metadata
file is allowed; plots and estimates are still produced, but truth-matched error
fields are omitted. When metadata exists, its number of coordinates must match
`--num_sources`.

`--txt` cannot be combined with `--glob`, because one metadata path cannot
describe multiple recordings.

## Quick start

### One recording

```powershell
python .\localize.py `
  --csv .\microphone_positions.csv `
  --wav .\data\01-001.wav `
  --txt .\data\01-001.txt `
  --num_sources 1 `
  --outdir .\out
```

### Multiple recordings

All files matched by `--glob` must have the same known source count:

```powershell
python .\localize.py `
  --csv .\microphone_positions.csv `
  --glob "data/03-*.wav" `
  --num_sources 3 `
  --outdir .\out_three_source
```

### Skip MVDR/Capon

MVDR/Capon can be omitted when only SRP-PHAT and DAS/Bartlett are needed:

```powershell
python .\localize.py `
  --csv .\microphone_positions.csv `
  --wav .\data\01-001.wav `
  --num_sources 1 `
  --skip_mvdr
```

Run `python .\localize.py --help` for the complete command-line help.

## Command-line options

| Option | Default | Description |
|---|---:|---|
| `--csv` | `microphone_positions.csv` | Microphone-coordinate CSV. |
| `--wav` | none | Process one WAV file. Exactly one of `--wav` and `--glob` is required. |
| `--glob` | none | Process WAV files matching a glob, in sorted path order. |
| `--txt` | matching WAV stem | Metadata for a single WAV; incompatible with `--glob`. |
| `--outdir` | `out` | Output directory. It is created when missing. |
| `--num_sources` | required | Known source count: `1` or `3`. |
| `--mics_use` | `32` | Number of microphones to use; valid range is 2 through the available count. |
| `--mic_selection` | `farthest` | `farthest` for deterministic geometric farthest-point sampling or `id` for approximately uniform numeric-ID selection. |
| `--nperseg` | `1024` | STFT window length in samples. |
| `--noverlap` | `768` | STFT overlap in samples; must be smaller than `nperseg`. |
| `--fmin` | `1000` | Lowest retained STFT frequency in hertz. |
| `--fmax` | `16000` | Highest retained STFT frequency in hertz. |
| `--active_quantile` | `0.70` | Frame-power quantile threshold in `[0, 1)`; `0` retains all frames at or above the minimum power. |
| `--xmin`, `--xmax` | `2.0`, `5.0` | Search bounds along x in metres. |
| `--ymin`, `--ymax` | `0.2`, `3.6` | Search bounds along y in metres. |
| `--zmin`, `--zmax` | `0.0`, `2.8` | Search bounds along z in metres. |
| `--step` | `0.1` | Cartesian grid spacing in metres. |
| `--min_sep` | `0.2` | Minimum Euclidean separation between selected estimates in metres. |
| `--speed_of_sound` | `343.0` | Propagation speed in metres per second. |
| `--mvdr_dl` | `0.01` | MVDR diagonal-loading factor. |
| `--batch` | `128` | Candidate positions evaluated per score-map batch. |
| `--skip_mvdr` | off | Do not compute MVDR/Capon. |

## Processing pipeline

1. **Load geometry.** The array centre is added to every relative microphone
   coordinate. Microphones are numerically ordered and a deterministic subset is
   selected.
2. **Load audio.** The selected channel indices are applied to both the audio and
   coordinate arrays.
3. **Compute the multichannel STFT.** SciPy's STFT is evaluated independently for
   each selected channel with no boundary extension.
4. **Select frequencies and frames.** Frequency bins inside the configured band
   are retained. Frame power is averaged over selected microphones and frequency
   bins, then thresholded at `active_quantile`.
5. **Build the grid.** The grid has shape `(number of y values, number of x
   values, number of z values)` and is flattened in C order, with z changing
   fastest.
6. **Evaluate score maps.** All methods use candidate-to-microphone propagation
   delays. DAS/Bartlett and MVDR/Capon use phase-only steering vectors based on
   delays relative to selected microphone index 0; no `1/r` amplitude term is
   included.
7. **Extract regional maxima.** Equal-valued voxels connected by a face, edge, or
   corner form one plateau. A plateau is a regional maximum only when no valid
   26-neighbour has a larger score. Its smallest flat grid index represents it.
8. **Select exactly K peaks.** Candidates are ordered by descending score and then
   ascending flat index. For `K=1`, the highest candidate is returned. For `K=3`,
   an exact branch-and-bound search finds the feasible triple with the greatest
   total score while enforcing `min_sep`. The selector fails clearly if no such
   set exists; it does not create fallback estimates.
9. **Evaluate against truth.** When metadata is present, Hungarian one-to-one
   matching reorders results by original truth index. It reports 3-D, depth
   (`|delta x|`), and in-plane (`sqrt(delta y^2 + delta z^2)`) errors, plus the
   nearest-grid error baseline.

## Localization methods

### SRP-PHAT

For every microphone pair, the implementation forms `X_i * conj(X_j)`, applies
PHAT normalization per time frame, averages over selected frames, compensates
the candidate-dependent pair delay, and sums the real contribution over pairs
and retained frequencies.

### DAS/Bartlett

The uncentered spatial second-moment matrix is calculated as `R = X X^H / T`.
For each candidate and retained frequency, the Bartlett quadratic form
`a^H R a` is evaluated and its real part is accumulated.

### MVDR/Capon

The same selected STFT frames define the spatial second-moment matrix. Each
frequency matrix receives trace-scaled diagonal loading. The implementation
solves the loaded linear system and sums `1 / real(a^H R_loaded^-1 a)` across
frequencies, with numerical floors where needed.

## Outputs

For every processed recording and computed method (`srp`, `das`, and optionally
`mvdr`), the output directory contains:

- `<recording>_<method>_xy_max_z.png` -- x-y maximum projection over z;
- `<recording>_<method>_xz_max_y.png` -- x-z maximum projection over y;
- `<recording>_<method>_yz_max_x.png` -- y-z maximum projection over x;
- `<recording>_<method>_z_<height>.png` -- normalized x-y slices at truth heights,
  or at estimated heights when truth is unavailable;
- `results.csv` -- one row per completed recording.

The CSV always records the run configuration and score-ranked raw estimates.
With truth metadata it also records:

- one-based truth and estimate indices after matching;
- truth-indexed estimated and true coordinates;
- per-source 3-D, depth, and in-plane errors;
- mean, median, and maximum matched errors;
- per-source and aggregate nearest-grid errors.

`results.csv` is rewritten after each completed recording, so results from
earlier recordings remain available if a later run is interrupted. Existing
rows are not automatically loaded or skipped; rerunning the same command starts
a new in-memory result table.

## Determinism and failure behavior

Microphone selection, file ordering, regional-maxima ordering, plateau
representation, score ties, and fixed-count selection are deterministic for
identical inputs. The program raises an explicit error for conditions including:

- an invalid or missing input selection;
- incompatible `--txt` and `--glob` arguments;
- a microphone count outside the available range;
- channel/coordinate count mismatch;
- metadata/source-count mismatch;
- an empty frequency band or activity mask;
- non-finite score maps; or
- too few mutually separated regional maxima for the requested source count.

## Scope and limitations

- The implementation performs an exhaustive grid search, so finer grids and
  larger search volumes increase runtime and memory use.
- Steering is geometric and phase-only. It does not model distance-dependent
  amplitude, measured room transfer functions, reverberation, or microphone
  calibration differences.
- Source-count estimation is outside the program's scope.
- Grid spacing limits coordinate resolution, and `min_sep` can prevent close
  sources from being selected as distinct estimates.
- The provided runtime is a research implementation, not an optimized real-time
  system.
