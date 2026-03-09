# Microphone Array Bird Localization (SRP-PHAT / DAS / MVDR)

This repository contains a baseline localization pipeline for simulated multi-channel bird song recordings captured by a planar microphone array. The script:

- loads a multichannel `.wav` recording (112 channels)
- loads microphone positions from `microphone_positions.csv`
- optionally loads per-recording ground truth from a `.txt` metadata file
- computes wideband spatial spectra over a 3D grid using:
  - **SRP-PHAT** (robust localization in reverberant environments)
  - **DAS** (conventional beamformer power spectrum; baseline)
  - **MVDR/Capon** (minimum variance distortionless response; sharper than DAS)
- estimates multiple sources by **peak picking** (non-maximum suppression)
- saves plots overlaying **estimated peaks** and **truth positions**
- writes `results.csv` with top peak coordinates and scores

> Note: This pipeline uses a **near-field point-source model** by scanning candidate 3D positions \((x,y,z)\) (not just directions).

---

## Files

Expected files in the working directory:

- `microphone_positions.csv`  
  Contains the microphone geometry:
  - one row labeled `Centre` with absolute array center coordinates (room frame, meters)
  - microphone rows `001`..`112` with coordinates **relative to the centre**

- `<name>.wav`  
  Multichannel recording:
  - sample rate typically **48 kHz**
  - **112 channels**
  - **channel order matches mic IDs** `001..112`

- `<name>.txt` (optional)  
  Metadata file containing ground truth source coordinates; the script extracts all `[x, y, z]` triplets.

Outputs are written to `out/` by default:
- `<name>_srp_slice.png`
- `<name>_das_slice.png`
- `<name>_mvdr_slice.png`
- `results.csv`

---

## Setup

### Python version
- Python 3.9+ recommended

### Install dependencies

```bash
pip install numpy pandas scipy soundfile matplotlib
```

## Usage

### Run on a single file

```bash
python localize.py --wav path_to_wav
```abs
### Run on all wavs
```bash
python localize.py --glob "path_to_wavs/*.wav"
```

## What the script does (pipeline overview)

### 1) Load geometry
Reads `microphone_positions.csv` and converts relative microphone coordinates to absolute room coordinates:

### 2) Load multichannel WAV
Loads the multichannel recording into an array of shape:

- `(samples, 112)`

Channel index `0` corresponds to mic `001`, channel index `111` to mic `112` (assuming channel order matches mic IDs).

### 3) STFT per microphone
Computes a Short-Time Fourier Transform (STFT) for each channel:

- `Z[m, f, t]` is the complex STFT coefficient for microphone `m`,
  frequency bin `f`, and time frame `t`.

This yields:
- `freqs`: array of frequency bin centers (Hz)
- `times`: array of frame times (seconds)
- `Z`: complex STFT data for all microphones

### 4) Frequency selection
Selects a frequency band (defaults: 1–9 kHz) to focus on bird vocalization content and reduce low-frequency noise.

### 5) Activity mask (frame selection)
Computes a simple energy measure per STFT frame (averaged across microphones and selected frequency bins) and keeps only the most energetic frames (default: top ~30%). This stabilizes spatial statistics by emphasizing frames where birds are active.

### 6) Build a 3D search grid (near-field)
Creates a 3D grid of candidate source locations \((x,y,z)\) within user-defined bounds and resolution (`step`). This is a near-field approach (localization by scanning points, not only directions).

### 7) Compute spatial spectra (scoring each grid point)
For each candidate location, the script computes wideband spatial scores over the chosen frequency band:

- **SRP-PHAT**: uses PHAT-normalized pairwise cross-spectra to measure phase/time-delay consistency with the candidate point.
- **DAS**: conventional beamformer (Bartlett) power, computed as \(\mathbf{a}^H\mathbf{R}\mathbf{a}\) per frequency and summed over frequencies.
- **MVDR/Capon**: minimum-variance distortionless response spectrum, computed as \(1/(\mathbf{a}^H\mathbf{R}^{-1}\mathbf{a})\) per frequency and summed over frequencies (with diagonal loading).

All methods use a near-field steering model based on distances from the candidate point to each microphone.

### 8) Peak picking (multi-source estimate)
Finds multiple candidate sources by selecting local maxima in the spatial map via non-maximum suppression:
- keeps peaks above `rel_thresh * max_peak`
- enforces a minimum separation `min_sep` between peaks
- returns up to `max_peaks`

The number of returned peaks is treated as an estimate of the number of sources in the recording.

### 9) Plotting and evaluation (optional)
For each method, the script saves an x–y heatmap slice at the z-value of the strongest estimated peak, and overlays:
- estimated peak locations (x markers)
- truth positions (o markers, if a `.txt` file is available)

If truth is available, it also computes and prints the minimum Euclidean distance from the top estimated peak to any truth position.
