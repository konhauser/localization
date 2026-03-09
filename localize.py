#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft
import matplotlib.pyplot as plt


# ---------------------------
# IO helpers
# ---------------------------

def _to_float(x):
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    s = str(x).strip()
    s = re.sub(r'[\[\]\s]', '', s)  # remove brackets/spaces
    return float(s)

def load_mic_positions(csv_path: str):
    """
    CSV format:
      - one row "Centre" with absolute coords in room frame
      - rows 001..112 with coords relative to the centre
    Returns: center (3,), abs_pos (M,3) ordered by mic ID
    """
    df = pd.read_csv(csv_path)

    centre_row = df[df["mic_id"].astype(str).str.lower() == "centre"].iloc[0]
    center = np.array([_to_float(centre_row["x"]),
                       _to_float(centre_row["y"]),
                       _to_float(centre_row["z"])], dtype=float)

    m = df[df["mic_id"].astype(str).str.match(r"^\d+$")].copy()
    m["id_int"] = m["mic_id"].astype(int)
    m = m.sort_values("id_int")

    rel = np.stack(
        [m["x"].map(_to_float).to_numpy(),
         m["y"].map(_to_float).to_numpy(),
         m["z"].map(_to_float).to_numpy()],
        axis=1
    )
    abs_pos = rel + center
    return center, abs_pos

def parse_truth_xyz(txt_path: str):
    """
    Extract all occurrences of [x, y, z] triplets from the metadata .txt.
    Returns: (Nsources, 3) array.
    """
    xyz = []
    with open(txt_path, "r") as f:
        for line in f:
            m = re.search(r"\[([0-9\.\-eE]+),\s*([0-9\.\-eE]+),\s*([0-9\.\-eE]+)\]", line)
            if m:
                xyz.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    return np.array(xyz, dtype=float)


# ---------------------------
# Geometry / grids
# ---------------------------

def make_grid(xrng, yrng, zrng, step):
    xs = np.arange(xrng[0], xrng[1] + 1e-9, step)
    ys = np.arange(yrng[0], yrng[1] + 1e-9, step)
    zs = np.arange(zrng[0], zrng[1] + 1e-9, step)

    # meshgrid with indexing="xy" gives arrays shaped (ny, nx, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    shape = (len(ys), len(xs), len(zs))  # (ny, nx, nz)
    return pts, xs, ys, zs, shape

def choose_mics(abs_pos, mics_use: int):
    """
    Choose a subset of microphones for speed (here default is all 112).
    Keeps a wide aperture by spacing indices.
    """
    M = abs_pos.shape[0]
    if mics_use >= M:
        return np.arange(M, dtype=int)
    idx = np.unique(np.round(np.linspace(0, M - 1, mics_use)).astype(int))
    return idx

def compute_tau_rel_grid(abs_pos, grid_pts, c=343.0, ref_idx=0):
    """
    Precompute near-field relative delays (vs reference mic) for every grid point.
    Returns tau_rel: (G,M) in seconds
    """
    # dist: (G,M)
    dist = np.linalg.norm(abs_pos[None, :, :] - grid_pts[:, None, :], axis=2)
    tau_rel = (dist - dist[:, [ref_idx]]) / c
    return tau_rel.astype(np.float64)


# ---------------------------
# STFT + masking
# ---------------------------

def stft_multich(x, sr, nperseg=1024, noverlap=768):
    """
    x: (samples, M)
    returns: freqs (F,), times (T,), Z (M,F,T) complex64
    """
    freqs, times, Z0 = stft(x[:, 0], fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    Z = np.empty((x.shape[1], len(freqs), len(times)), dtype=np.complex64)
    Z[0] = Z0.astype(np.complex64)
    for m in range(1, x.shape[1]):
        _, _, Zm = stft(x[:, m], fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
        Z[m] = Zm.astype(np.complex64)
    return freqs, times, Z

def activity_mask(Z, fmask, quantile_keep=0.70):
    """
    Keep "bird-active" frames using energy in a selected frequency band.
    quantile_keep=0.70 => keep top 30% energy frames.
    """
    P = np.mean(np.abs(Z[:, fmask, :]) ** 2, axis=(0, 1))  # (T,)
    thr = np.quantile(P, quantile_keep)
    return P >= thr


# ---------------------------
# Peak picking = source-count estimate
# ---------------------------

def nms_peaks(grid_pts, scores, min_sep=0.6, rel_thresh=0.65, max_peaks=10):
    """
    Non-maximum suppression:
      - keep peaks above rel_thresh * best_score
      - enforce spatial separation >= min_sep (meters)
    Returns list of kept indices.
    """
    idx = np.argsort(scores)[::-1]
    best = idx[0]
    smax = scores[best]
    keep = []
    for k in idx:
        if scores[k] < rel_thresh * smax:
            break
        p = grid_pts[k]
        if all(np.linalg.norm(p - grid_pts[j]) >= min_sep for j in keep):
            keep.append(k)
        if len(keep) >= max_peaks:
            break
    return keep


# ---------------------------
# Near-field scoring
# ---------------------------

def srp_phat_score(abs_pos, Z, freqs, fmask, tmask, grid_pts, c=343.0, batch=256):
    """
    Near-field SRP-PHAT:
      score(s) = sum_{pairs,f} Re{ Gmn(f) * exp(j*2pi f * (tau_m(s)-tau_n(s))) }
    where Gmn(f) is PHAT-normalized cross-spectrum averaged over active frames.
    """
    M = abs_pos.shape[0]
    use_freqs = freqs[fmask].astype(np.float64)  # (K,)
    X = Z[:, fmask, :][:, :, tmask]              # (M,K,Tact)
    K = X.shape[1]

    # mic pairs
    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
    Pn = len(pairs)

    # PHAT cross-spectrum averaged over time
    Gmn = np.empty((Pn, K), dtype=np.complex64)
    for p, (i, j) in enumerate(pairs):
        C = X[i] * np.conj(X[j])          # (K,Tact)
        C /= (np.abs(C) + 1e-12)          # PHAT
        Gmn[p] = np.mean(C, axis=1)       # (K,)

    w = (2.0 * np.pi * use_freqs).astype(np.float64)  # (K,)

    scores = np.zeros((grid_pts.shape[0],), dtype=np.float64)

    for start in range(0, grid_pts.shape[0], batch):
        end = min(start + batch, grid_pts.shape[0])
        S = grid_pts[start:end]  # (B,3)

        # distances -> delays
        Rdist = np.linalg.norm(abs_pos[None, :, :] - S[:, None, :], axis=2)  # (B,M)
        Tau = Rdist / c                                                      # (B,M)

        # dtau for each pair
        dtau = np.empty((S.shape[0], Pn), dtype=np.float64)
        for p, (i, j) in enumerate(pairs):
            dtau[:, p] = Tau[:, i] - Tau[:, j]

        # exp(+j*w*dtau): (B,Pn,K)
        phase = np.exp(1j * dtau[:, :, None] * w[None, None, :]).astype(np.complex64)

        # sum real part
        sc = np.real(np.sum(phase * Gmn[None, :, :], axis=(1, 2)))
        scores[start:end] = sc

    return scores

def das_score_from_tau_rel(Z, freqs, fmask, tmask, tau_rel_grid, batch=256):
    """
    Near-field DAS power map using precomputed tau_rel_grid (G,M):
      P_DAS(s) = sum_k a(s,f_k)^H R(f_k) a(s,f_k)
    """
    # X: (M,K,Tact)
    X = Z[:, fmask, :][:, :, tmask]
    M = X.shape[0]
    K = X.shape[1]
    Tact = X.shape[2]
    use_freqs = freqs[fmask].astype(np.float64)

    # Covariance per frequency bin: R_k (K,M,M)
    R = np.empty((K, M, M), dtype=np.complex64)
    for k in range(K):
        Xk = X[:, k, :]  # (M,Tact)
        R[k] = (Xk @ np.conj(Xk.T)) / max(Tact, 1)

    G = tau_rel_grid.shape[0]
    scores = np.zeros((G,), dtype=np.float64)

    for start in range(0, G, batch):
        end = min(start + batch, G)
        tau_b = tau_rel_grid[start:end]  # (B,M)
        sc = np.zeros((end - start,), dtype=np.float64)

        for k in range(K):
            a = np.exp(-1j * 2*np.pi * use_freqs[k] * tau_b).astype(np.complex64)  # (B,M)
            Ra = a @ R[k]                                                         # (B,M)
            sc += np.real(np.sum(np.conj(a) * Ra, axis=1))

        scores[start:end] = sc

    return scores

def mvdr_score_from_tau_rel(Z, freqs, fmask, tmask, tau_rel_grid, dl_factor=1e-2, batch=256):
    """
    Near-field MVDR/Capon spatial spectrum (wideband) using precomputed tau_rel_grid (G,M):
      P_MVDR(s) = sum_k 1 / (a^H R^{-1} a)
    with diagonal loading:
      R_loaded = R + dl * I, dl = dl_factor * tr(R).real / M
    """
    X = Z[:, fmask, :][:, :, tmask]  # (M,K,Tact)
    M = X.shape[0]
    K = X.shape[1]
    Tact = X.shape[2]
    use_freqs = freqs[fmask].astype(np.float64)

    G = tau_rel_grid.shape[0]
    scores = np.zeros((G,), dtype=np.float64)

    I = np.eye(M, dtype=np.complex128)

    for k in range(K):
        Xk = X[:, k, :]  # (M,Tact)
        Rk = (Xk @ np.conj(Xk.T)) / max(Tact, 1)  # (M,M) complex
        Rk = Rk.astype(np.complex128)

        dl = float(dl_factor) * (np.trace(Rk).real / M)
        Rk_loaded = Rk + dl * I

        # Process grid points in batches (solve with multiple RHS for speed)
        for start in range(0, G, batch):
            end = min(start + batch, G)
            tau_b = tau_rel_grid[start:end]  # (B,M)

            a = np.exp(-1j * 2*np.pi * use_freqs[k] * tau_b).astype(np.complex128)  # (B,M)
            # Solve Rk_loaded * y = a^T  => y = R^{-1} a^T  (M,B)
            y = np.linalg.solve(Rk_loaded, a.T)  # (M,B)

            # denom = a^H R^{-1} a  => for each b: sum_m conj(a[b,m]) * y[m,b]
            denom = np.sum(np.conj(a) * y.T, axis=1)  # (B,)

            # Capon spectrum is 1/denom (denom should be real positive)
            denom_real = np.maximum(np.real(denom), 1e-12)
            scores[start:end] += 1.0 / denom_real

    return scores


# ---------------------------
# Plot helper: x-y slice at z of strongest peak + truth overlay
# ---------------------------

def save_slice_plot(scores, grid_shape, xs, ys, zs, peaks, out_png, title,
                    truth=None, dz_tol=None):
    """
    Plot x-y slice at the z of the strongest estimated peak, mark estimated peaks,
    and overlay truth positions if provided.

    truth: array shape (Ntruth,3) in room coords
    dz_tol: if not None, only show truth points with |z_truth - z_slice| <= dz_tol
    """
    ny, nx, nz = grid_shape
    S3 = scores.reshape((ny, nx, nz))

    best = peaks[0]
    iy, ix, iz = np.unravel_index(best, (ny, nx, nz))
    z0 = zs[iz]
    plane = S3[:, :, iz]  # (ny,nx)

    plt.figure(figsize=(8, 4))
    plt.imshow(
        plane, origin="lower", aspect="auto",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()]
    )
    plt.colorbar(label="score")

    # Estimated peaks
    for i, k in enumerate(peaks, 1):
        iy2, ix2, iz2 = np.unravel_index(k, (ny, nx, nz))
        plt.scatter(xs[ix2], ys[iy2], marker="x", label="estimate" if i == 1 else None)

    # Truth overlay
    if truth is not None and truth.size > 0:
        truth_show = truth
        if dz_tol is not None:
            truth_show = truth[np.abs(truth[:, 2] - z0) <= dz_tol]

        if truth_show.size > 0:
            plt.scatter(truth_show[:, 0], truth_show[:, 1], marker="o", label="truth")
            for i, p in enumerate(truth_show, 1):
                plt.text(p[0], p[1], f"T{i}", fontsize=8)
        elif dz_tol is not None:
            plt.text(xs.min(), ys.max(), f"(no truth within dz≤{dz_tol}m)", fontsize=8)

    plt.title(f"{title} (z-slice≈{z0:.2f} m)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="microphone_positions.csv", help="mic positions CSV")
    ap.add_argument("--wav", default=None, help="single wav to process")
    ap.add_argument("--glob", default=None, help='process many wavs, e.g. "*.wav"')
    ap.add_argument("--txt", default=None, help="matching txt (optional); if omitted uses wav stem + .txt")

    ap.add_argument("--outdir", default="out", help="output directory")

    # Always use all 112 by default
    ap.add_argument("--mics_use", type=int, default=32, help="how many mics to use (default: all 112)")

    ap.add_argument("--nperseg", type=int, default=1024)
    ap.add_argument("--noverlap", type=int, default=768)

    ap.add_argument("--fmin", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=16000.0)
    ap.add_argument("--active_quantile", type=float, default=0.7, help="keep frames above this energy quantile")

    # grid (your current defaults)
    ap.add_argument("--xmin", type=float, default=2.0, help="if None, uses center_x + 0.2")
    ap.add_argument("--xmax", type=float, default=5.0)
    ap.add_argument("--ymin", type=float, default=0.2)
    ap.add_argument("--ymax", type=float, default=3.6)
    ap.add_argument("--zmin", type=float, default=0.0)
    ap.add_argument("--zmax", type=float, default=2.8)
    ap.add_argument("--step", type=float, default=0.25, help="grid step in meters")

    # peak picking / source-count estimate
    ap.add_argument("--min_sep", type=float, default=0.15, help="min separation between sources (m)")
    ap.add_argument("--rel_thresh", type=float, default=0.65, help="peak threshold relative to best peak")
    ap.add_argument("--max_peaks", type=int, default=6)

    # plotting truth
    ap.add_argument("--truth_dz_tol", type=float, default=None,
                    help="Only plot truth points within +/- this z distance of the plotted z-slice (meters). Default: show all truths.")

    # MVDR parameters
    ap.add_argument("--mvdr_dl", type=float, default=1e-2,
                    help="MVDR diagonal loading factor: dl = mvdr_dl * trace(R)/M")
    ap.add_argument("--skip_mvdr", action="store_true",
                    help="Skip MVDR computation (useful if it's too slow).")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    center, abs_pos_all = load_mic_positions(args.csv)
    print("Array center (room coords):", center)
    print("Total mics:", abs_pos_all.shape[0])

    # which wavs?
    if args.glob:
        wav_paths = sorted(Path(".").glob(args.glob))
    else:
        if not args.wav:
            raise SystemExit("Provide --wav FILE.wav or --glob '*.wav'")
        wav_paths = [Path(args.wav)]

    rows = []

    for wav_path in wav_paths:
        wav_path = Path(wav_path)
        stem = wav_path.stem
        txt_path = Path(args.txt) if args.txt else wav_path.with_suffix(".txt")
        has_truth = txt_path.exists()

        # Load truth once per file and print it
        truth = None
        print("\n--- Processing:", wav_path.name, "---")
        if has_truth:
            truth = parse_truth_xyz(str(txt_path))
            print("Truth positions:", truth.shape[0])
            for i, p in enumerate(truth, 1):
                print(f"  truth {i}: x={p[0]:.4f}, y={p[1]:.4f}, z={p[2]:.4f}")
        else:
            print("Truth positions: (no .txt found)")

        x, sr = sf.read(str(wav_path), always_2d=True)
        print("Audio shape:", x.shape, "sr:", sr)
        if x.shape[1] != abs_pos_all.shape[0]:
            raise RuntimeError(f"Channel count {x.shape[1]} != mic count {abs_pos_all.shape[0]}")

        mic_idx = choose_mics(abs_pos_all, args.mics_use)
        abs_pos = abs_pos_all[mic_idx]
        x_use = x[:, mic_idx]
        print("Using mics:", len(mic_idx))

        freqs, times, Z = stft_multich(x_use, sr, nperseg=args.nperseg, noverlap=args.noverlap)
        fmask = (freqs >= args.fmin) & (freqs <= args.fmax)
        tmask = activity_mask(Z, fmask, quantile_keep=args.active_quantile)
        print("Active frames:", int(np.sum(tmask)), "/", len(tmask), f"({np.mean(tmask)*100:.1f}%)")

        # grid
        xmin = args.xmin if args.xmin is not None else (center[0] + 0.2)
        grid, xs, ys, zs, shape = make_grid((xmin, args.xmax),
                                            (args.ymin, args.ymax),
                                            (args.zmin, args.zmax),
                                            args.step)
        ny, nx, nz = shape
        print("Grid:", grid.shape[0], "points",
              f"(nx={nx}, ny={ny}, nz={nz}, step={args.step} m)")

        # Debug: SRP score at nearest grid point to each truth
        # Compute SRP first (needed for this debug).
        srp = srp_phat_score(abs_pos, Z, freqs, fmask, tmask, grid, c=343.0, batch=256)

        if truth is not None and truth.size > 0:
            srp_max = float(np.max(srp))
            print("SRP score at nearest grid point to each truth:")
            for i, p in enumerate(truth, 1):
                k = int(np.argmin(np.sum((grid - p[None, :])**2, axis=1)))
                print(f"  truth {i}: nearest_grid={grid[k]}  srp={srp[k]:.2f}  ratio={srp[k]/srp_max:.3f}")

        peaks_srp = nms_peaks(grid, srp, min_sep=args.min_sep, rel_thresh=args.rel_thresh, max_peaks=args.max_peaks)
        print("SRP-PHAT estimated #sources:", len(peaks_srp))
        for i, k in enumerate(peaks_srp, 1):
            print(f"  SRP peak {i}: {grid[k]}  score={srp[k]:.2f}")

        # Precompute tau_rel for DAS/MVDR (faster than recomputing distances inside every freq loop)
        tau_rel_grid = compute_tau_rel_grid(abs_pos, grid, c=343.0, ref_idx=0)

        # DAS
        das = das_score_from_tau_rel(Z, freqs, fmask, tmask, tau_rel_grid, batch=256)
        peaks_das = nms_peaks(grid, das, min_sep=args.min_sep, rel_thresh=args.rel_thresh, max_peaks=args.max_peaks)
        print("DAS estimated #sources:", len(peaks_das))
        for i, k in enumerate(peaks_das, 1):
            print(f"  DAS peak {i}: {grid[k]}  score={das[k]:.2f}")

        # MVDR
        mvdr = None
        peaks_mvdr = []
        if not args.skip_mvdr:
            mvdr = mvdr_score_from_tau_rel(
                Z, freqs, fmask, tmask, tau_rel_grid,
                dl_factor=args.mvdr_dl,
                batch=256
            )
            peaks_mvdr = nms_peaks(grid, mvdr, min_sep=args.min_sep, rel_thresh=args.rel_thresh, max_peaks=args.max_peaks)
            print("MVDR estimated #sources:", len(peaks_mvdr))
            for i, k in enumerate(peaks_mvdr, 1):
                print(f"  MVDR peak {i}: {grid[k]}  score={mvdr[k]:.2f}")

        # plots (overlay truth)
        srp_png = outdir / f"{stem}_srp_slice.png"
        das_png = outdir / f"{stem}_das_slice.png"
        save_slice_plot(srp, shape, xs, ys, zs, peaks_srp, srp_png, "SRP-PHAT",
                        truth=truth, dz_tol=args.truth_dz_tol)
        save_slice_plot(das, shape, xs, ys, zs, peaks_das, das_png, "DAS",
                        truth=truth, dz_tol=args.truth_dz_tol)

        print("Wrote:", srp_png, "and", das_png)

        mvdr_png = None
        if mvdr is not None and len(peaks_mvdr) > 0:
            mvdr_png = outdir / f"{stem}_mvdr_slice.png"
            save_slice_plot(mvdr, shape, xs, ys, zs, peaks_mvdr, mvdr_png, "MVDR",
                            truth=truth, dz_tol=args.truth_dz_tol)
            print("Wrote:", mvdr_png)

        # truth (optional): quick “min distance to any truth” for the top peak of each method
        def top1_min_dist(peaks, truth_arr):
            if truth_arr is None or truth_arr.size == 0 or len(peaks) == 0:
                return np.nan
            return float(np.min(np.linalg.norm(truth_arr - grid[peaks[0]][None, :], axis=1)))

        srp_top1_err = top1_min_dist(peaks_srp, truth)
        das_top1_err = top1_min_dist(peaks_das, truth)
        mvdr_top1_err = top1_min_dist(peaks_mvdr, truth) if mvdr is not None else np.nan

        if truth is not None and truth.size > 0:
            print("Truth count:", truth.shape[0],
                  "\nSRP top1 min-dist:", srp_top1_err,
                  "\nDAS top1 min-dist:", das_top1_err,
                  ("\nMVDR top1 min-dist: " + str(mvdr_top1_err)) if mvdr is not None else "")

        # pack top 3 peaks for csv
        def pack3(peaks, scores):
            out = []
            for k in peaks[:3]:
                out.append((grid[k][0], grid[k][1], grid[k][2], float(scores[k])))
            while len(out) < 3:
                out.append((np.nan, np.nan, np.nan, np.nan))
            return out

        srp3 = pack3(peaks_srp, srp)
        das3 = pack3(peaks_das, das)
        mvdr3 = pack3(peaks_mvdr, mvdr) if mvdr is not None else [(np.nan, np.nan, np.nan, np.nan)] * 3

        rows.append({
            "file": wav_path.name,
            "sr": sr,
            "channels": x.shape[1],
            "mics_used": len(mic_idx),

            "srp_N": len(peaks_srp),
            "srp_x1": srp3[0][0], "srp_y1": srp3[0][1], "srp_z1": srp3[0][2], "srp_score1": srp3[0][3],
            "srp_x2": srp3[1][0], "srp_y2": srp3[1][1], "srp_z2": srp3[1][2], "srp_score2": srp3[1][3],
            "srp_x3": srp3[2][0], "srp_y3": srp3[2][1], "srp_z3": srp3[2][2], "srp_score3": srp3[2][3],

            "das_N": len(peaks_das),
            "das_x1": das3[0][0], "das_y1": das3[0][1], "das_z1": das3[0][2], "das_score1": das3[0][3],
            "das_x2": das3[1][0], "das_y2": das3[1][1], "das_z2": das3[1][2], "das_score2": das3[1][3],
            "das_x3": das3[2][0], "das_y3": das3[2][1], "das_z3": das3[2][2], "das_score3": das3[2][3],

            "mvdr_N": len(peaks_mvdr) if mvdr is not None else np.nan,
            "mvdr_x1": mvdr3[0][0], "mvdr_y1": mvdr3[0][1], "mvdr_z1": mvdr3[0][2], "mvdr_score1": mvdr3[0][3],
            "mvdr_x2": mvdr3[1][0], "mvdr_y2": mvdr3[1][1], "mvdr_z2": mvdr3[1][2], "mvdr_score2": mvdr3[1][3],
            "mvdr_x3": mvdr3[2][0], "mvdr_y3": mvdr3[2][1], "mvdr_z3": mvdr3[2][2], "mvdr_score3": mvdr3[2][3],

            "truth_count": truth.shape[0] if truth is not None else np.nan,
            "srp_top1_min_dist_to_truth": srp_top1_err,
            "das_top1_min_dist_to_truth": das_top1_err,
            "mvdr_top1_min_dist_to_truth": mvdr_top1_err,
        })

    out_csv = outdir / "results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("\nSaved summary:", out_csv)


if __name__ == "__main__":
    main()