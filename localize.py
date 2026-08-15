#!/usr/bin/env python3
"""Near-field localization for simulated multichannel bird recordings."""

from __future__ import annotations

import argparse
import re
from math import fsum
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import linear_sum_assignment
from scipy.signal import stft

EPS = 1e-12
PEAK_SEPARATION_TOLERANCE = 1e-12


# ---------------------------
# IO helpers
# ---------------------------

def _to_float(value: object) -> float:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    text = re.sub(r"[\[\]\s]", "", str(value).strip())
    return float(text)


def load_mic_positions(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the absolute array centre and absolute microphone positions.

    Expected CSV layout:
      - one row whose mic_id is ``centre`` with absolute room coordinates;
      - rows 001..112 with coordinates relative to the centre.
    """
    df = pd.read_csv(csv_path)
    required = {"mic_id", "x", "y", "z"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Microphone CSV is missing columns: {sorted(missing)}")

    centre_rows = df[df["mic_id"].astype(str).str.lower() == "centre"]
    if len(centre_rows) != 1:
        raise ValueError("Microphone CSV must contain exactly one 'centre' row.")

    centre_row = centre_rows.iloc[0]
    center = np.array(
        [_to_float(centre_row["x"]), _to_float(centre_row["y"]), _to_float(centre_row["z"])],
        dtype=np.float64,
    )

    microphones = df[df["mic_id"].astype(str).str.fullmatch(r"\d+")].copy()
    if microphones.empty:
        raise ValueError("Microphone CSV contains no numeric microphone rows.")

    microphones["id_int"] = microphones["mic_id"].astype(int)
    microphones = microphones.sort_values("id_int")
    relative = np.column_stack(
        [
            microphones["x"].map(_to_float).to_numpy(),
            microphones["y"].map(_to_float).to_numpy(),
            microphones["z"].map(_to_float).to_numpy(),
        ]
    ).astype(np.float64)

    return center, relative + center


def parse_truth_xyz(txt_path: str | Path) -> np.ndarray:
    """Extract all ``[x, y, z]`` coordinate triplets from a metadata file."""
    coordinates: list[list[float]] = []
    pattern = re.compile(
        r"\[\s*([0-9.\-+eE]+)\s*,\s*([0-9.\-+eE]+)\s*,\s*([0-9.\-+eE]+)\s*\]"
    )
    with Path(txt_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                coordinates.append([float(match.group(i)) for i in range(1, 4)])
    return np.asarray(coordinates, dtype=np.float64).reshape((-1, 3))


# ---------------------------
# Geometry and microphone selection
# ---------------------------

def make_grid(
    xrng: tuple[float, float],
    yrng: tuple[float, float],
    zrng: tuple[float, float],
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    for name, limits in (("x", xrng), ("y", yrng), ("z", zrng)):
        if limits[1] < limits[0]:
            raise ValueError(f"Invalid {name}-range: minimum exceeds maximum.")

    xs = np.arange(xrng[0], xrng[1] + step * 1e-6, step)
    ys = np.arange(yrng[0], yrng[1] + step * 1e-6, step)
    zs = np.arange(zrng[0], zrng[1] + step * 1e-6, step)
    x_grid, y_grid, z_grid = np.meshgrid(xs, ys, zs, indexing="xy")
    points = np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=1)
    shape = (len(ys), len(xs), len(zs))
    return points, xs, ys, zs, shape


def choose_mics(abs_pos: np.ndarray, count: int, mode: str = "farthest") -> np.ndarray:
    """Choose microphone indices.

    ``farthest`` uses deterministic farthest-point sampling in physical space,
    which tends to preserve array aperture. ``id`` reproduces the old selection
    based on approximately uniform microphone IDs.
    """
    total = abs_pos.shape[0]
    if not 2 <= count <= total:
        raise ValueError(
            f"mics_use must be between 2 and {total} inclusive; got {count}."
        )
    if count == total:
        return np.arange(total, dtype=int)

    if mode == "id":
        return np.unique(np.round(np.linspace(0, total - 1, count)).astype(int))
    if mode != "farthest":
        raise ValueError(f"Unknown microphone selection mode: {mode}")

    # All microphones are planar here, but using full 3D distances is harmless.
    center = np.mean(abs_pos, axis=0)
    first = int(np.argmax(np.linalg.norm(abs_pos - center, axis=1)))
    selected = [first]
    min_distance = np.linalg.norm(abs_pos - abs_pos[first], axis=1)

    while len(selected) < count:
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        min_distance = np.minimum(
            min_distance,
            np.linalg.norm(abs_pos - abs_pos[next_index], axis=1),
        )

    return np.sort(np.asarray(selected, dtype=int))


def compute_tau_rel_grid(
    abs_pos: np.ndarray,
    grid_pts: np.ndarray,
    speed_of_sound: float = 343.0,
    ref_idx: int = 0,
) -> np.ndarray:
    if speed_of_sound <= 0:
        raise ValueError("Speed of sound must be positive.")
    distances = np.linalg.norm(abs_pos[None, :, :] - grid_pts[:, None, :], axis=2)
    delays = distances / speed_of_sound
    return (delays - delays[:, [ref_idx]]).astype(np.float64)


# ---------------------------
# STFT and activity masking
# ---------------------------

def stft_multich(
    audio: np.ndarray,
    sample_rate: int,
    nperseg: int = 1024,
    noverlap: int = 768,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 <= noverlap < nperseg:
        raise ValueError("STFT overlap must satisfy 0 <= noverlap < nperseg.")

    frequencies, times, first = stft(
        audio[:, 0],
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
    )
    transformed = np.empty(
        (audio.shape[1], len(frequencies), len(times)), dtype=np.complex64
    )
    transformed[0] = first.astype(np.complex64)

    for mic in range(1, audio.shape[1]):
        _, _, channel = stft(
            audio[:, mic],
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None,
        )
        transformed[mic] = channel.astype(np.complex64)

    return frequencies, times, transformed


def activity_mask(
    transformed: np.ndarray,
    frequency_mask: np.ndarray,
    quantile_keep: float = 0.70,
) -> np.ndarray:
    if not 0.0 <= quantile_keep < 1.0:
        raise ValueError("active_quantile must be in [0, 1).")
    if not np.any(frequency_mask):
        raise ValueError("The selected frequency band contains no STFT bins.")

    frame_power = np.mean(
        np.abs(transformed[:, frequency_mask, :]) ** 2,
        axis=(0, 1),
    )
    threshold = np.quantile(frame_power, quantile_keep)
    mask = frame_power >= threshold
    if not np.any(mask):
        raise RuntimeError("Activity mask selected no time frames.")
    return mask


# ---------------------------
# Peak selection and evaluation
# ---------------------------

def _local_maximum_candidates(
    scores: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> list[int]:
    """Return deterministic representatives of 26-connected regional maxima.

    Equal-valued voxels connected through faces, edges, or corners form one
    plateau. A plateau is a regional maximum when none of its valid 26-neighbours
    has a greater score. Its smallest C-order flat index is the representative.
    """
    ny, nx, nz = grid_shape
    visited = np.zeros(scores.size, dtype=bool)
    maxima: list[int] = []
    xz_plane_size = nx * nz

    for start in range(scores.size):
        if visited[start]:
            continue

        plateau_value = scores[start]
        stack = [start]
        visited[start] = True
        is_regional_maximum = True

        while stack:
            current = stack.pop()
            iy, remainder = divmod(current, xz_plane_size)
            ix, iz = divmod(remainder, nz)

            for neighbour_y in range(max(0, iy - 1), min(ny, iy + 2)):
                for neighbour_x in range(max(0, ix - 1), min(nx, ix + 2)):
                    neighbour_base = (neighbour_y * nx + neighbour_x) * nz
                    for neighbour_z in range(max(0, iz - 1), min(nz, iz + 2)):
                        neighbour = neighbour_base + neighbour_z
                        if neighbour == current:
                            continue

                        neighbour_value = scores[neighbour]
                        if neighbour_value > plateau_value:
                            is_regional_maximum = False
                        elif neighbour_value == plateau_value and not visited[neighbour]:
                            visited[neighbour] = True
                            stack.append(neighbour)

        if is_regional_maximum:
            # The outer loop is ascending, so start is this plateau's smallest index.
            maxima.append(start)

    return sorted(maxima, key=lambda index: (-float(scores[index]), index))


def _separation_tolerance(min_separation: float) -> float:
    """Return the absolute geometric tolerance used at the separation boundary."""
    return PEAK_SEPARATION_TOLERANCE * max(1.0, min_separation)


def _forward_separation_masks(
    candidate_points: np.ndarray,
    minimum_distance_squared: float,
) -> list[int]:
    """Encode later compatible candidates as deterministic Python-int bitsets."""
    candidate_count = len(candidate_points)
    masks = [0] * candidate_count
    for left in range(candidate_count - 1):
        delta = candidate_points[left + 1 :] - candidate_points[left]
        distance_squared = np.einsum("ij,ij->i", delta, delta)
        compatible = distance_squared >= minimum_distance_squared
        packed = np.packbits(compatible, bitorder="little")
        masks[left] = int.from_bytes(packed.tobytes(), "little") << (left + 1)
    return masks


def nms_top_k_peaks(
    grid_pts: np.ndarray,
    scores: np.ndarray,
    grid_shape: tuple[int, int, int],
    number_of_sources: int,
    min_separation: float,
) -> list[int]:
    """Return the maximum-score feasible set of K regional maxima.

    Candidates are 26-connected regional maxima ordered by descending score and
    then ascending C-order flat index. For K=3, exact branch-and-bound search over
    compatible pairs finds the feasible triple with maximum total score. Returned
    peaks retain candidate order. Distances within
    ``1e-12 * max(1, min_separation)`` metres of the boundary are treated as
    satisfying ``min_separation``.
    """
    if number_of_sources not in (1, 3):
        raise ValueError("number_of_sources must be either 1 or 3.")
    if not np.isfinite(min_separation) or min_separation < 0:
        raise ValueError("min_separation must be finite and non-negative.")
    if scores.ndim != 1 or len(scores) != len(grid_pts):
        raise ValueError("scores must be one-dimensional and match the grid size.")
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Spatial score map contains NaN or infinite values.")
    if grid_pts.ndim != 2 or grid_pts.shape[1] != 3:
        raise ValueError("grid_pts must have shape (grid_size, 3).")
    if not np.all(np.isfinite(grid_pts)):
        raise ValueError("grid_pts contains NaN or infinite coordinates.")

    shape = tuple(int(length) for length in grid_shape)
    if len(shape) != 3 or any(length <= 0 for length in shape):
        raise ValueError("grid_shape must contain three positive dimensions.")
    if int(np.prod(shape, dtype=np.int64)) != len(scores):
        raise ValueError("grid_shape does not match the flat score-map size.")

    candidates = _local_maximum_candidates(scores, shape)
    candidate_count = len(candidates)
    if candidate_count < number_of_sources:
        raise RuntimeError(
            f"Could not select K={number_of_sources} peaks from {candidate_count} "
            f"local-maximum candidates with min_separation={min_separation} m."
        )
    if number_of_sources == 1:
        return [candidates[0]]

    candidate_points = grid_pts[candidates]
    candidate_scores = scores[candidates]
    tolerance = _separation_tolerance(min_separation)
    minimum_distance_squared = max(0.0, min_separation - tolerance) ** 2
    separation_masks = _forward_separation_masks(
        candidate_points,
        minimum_distance_squared,
    )

    best_combination: tuple[int, int, int] | None = None
    best_total_score = -np.inf

    # Candidate scores are non-increasing. The bounds ignore separation, so they
    # can only overestimate the best remaining total and are safe for exact search.
    for first in range(candidate_count - 2):
        first_upper_bound = fsum(
            float(candidate_scores[index])
            for index in (first, first + 1, first + 2)
        )
        if best_combination is not None and first_upper_bound <= best_total_score:
            break

        remaining_seconds = separation_masks[first]
        while remaining_seconds:
            second_bit = remaining_seconds & -remaining_seconds
            second = second_bit.bit_length() - 1
            remaining_seconds ^= second_bit
            if second >= candidate_count - 1:
                continue

            pair_upper_bound = fsum(
                float(candidate_scores[index])
                for index in (first, second, second + 1)
            )
            if best_combination is not None and pair_upper_bound <= best_total_score:
                break

            possible_thirds = separation_masks[first] & separation_masks[second]
            if not possible_thirds:
                continue

            third_bit = possible_thirds & -possible_thirds
            third = third_bit.bit_length() - 1
            total_score = fsum(
                float(candidate_scores[index])
                for index in (first, second, third)
            )
            if total_score > best_total_score:
                best_combination = (first, second, third)
                best_total_score = total_score

    if best_combination is None:
        raise RuntimeError(
            f"Could not select K={number_of_sources} mutually separated peaks from "
            f"{candidate_count} local-maximum candidates with "
            f"min_separation={min_separation} m."
        )

    return [candidates[index] for index in best_combination]


def match_estimates_to_truth(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Match estimates and return truth-ordered rows plus zero-based input indices."""
    if estimated.shape != truth.shape or estimated.ndim != 2 or estimated.shape[1] != 3:
        raise ValueError(
            f"Estimate/truth shape mismatch: estimated={estimated.shape}, truth={truth.shape}"
        )

    cost = np.linalg.norm(estimated[:, None, :] - truth[None, :, :], axis=2)
    estimate_indices, truth_indices = linear_sum_assignment(cost)
    truth_order = np.argsort(truth_indices)
    estimate_indices = estimate_indices[truth_order]
    truth_indices = truth_indices[truth_order]
    matched_estimated = estimated[estimate_indices]
    matched_truth = truth[truth_indices]
    delta = matched_estimated - matched_truth
    errors_3d = np.linalg.norm(delta, axis=1)

    # The microphone plane is y-z because all relative microphone x-values are zero.
    depth_error = np.abs(delta[:, 0])
    in_plane_error = np.linalg.norm(delta[:, 1:3], axis=1)

    return {
        "estimate_indices": estimate_indices,
        "truth_indices": truth_indices,
        "estimated": matched_estimated,
        "truth": matched_truth,
        "errors_3d": errors_3d,
        "depth_errors": depth_error,
        "in_plane_errors": in_plane_error,
        "mean_error": float(np.mean(errors_3d)),
        "median_error": float(np.median(errors_3d)),
        "max_error": float(np.max(errors_3d)),
    }


def nearest_grid_errors(grid_pts: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Minimum possible error caused solely by grid discretization."""
    return np.asarray(
        [np.min(np.linalg.norm(grid_pts - position[None, :], axis=1)) for position in truth],
        dtype=np.float64,
    )


# ---------------------------
# Near-field spatial spectra
# ---------------------------

def srp_phat_score(
    abs_pos: np.ndarray,
    transformed: np.ndarray,
    frequencies: np.ndarray,
    frequency_mask: np.ndarray,
    time_mask: np.ndarray,
    grid_pts: np.ndarray,
    speed_of_sound: float = 343.0,
    batch: int = 128,
) -> np.ndarray:
    """Compute a near-field SRP-PHAT score over the candidate grid."""
    mic_count = abs_pos.shape[0]
    selected_frequencies = frequencies[frequency_mask].astype(np.float64)
    x_selected = transformed[:, frequency_mask, :][:, :, time_mask]
    frequency_count = x_selected.shape[1]

    pair_i, pair_j = np.triu_indices(mic_count, k=1)
    pair_count = len(pair_i)
    if pair_count == 0:
        raise ValueError("SRP-PHAT requires at least two microphones.")

    cross_spectra = np.empty((pair_count, frequency_count), dtype=np.complex64)
    for pair_index, (left, right) in enumerate(zip(pair_i, pair_j, strict=True)):
        cross = x_selected[left] * np.conj(x_selected[right])
        cross /= np.abs(cross) + EPS
        cross_spectra[pair_index] = np.mean(cross, axis=1)

    angular_frequencies = 2.0 * np.pi * selected_frequencies
    scores = np.zeros(len(grid_pts), dtype=np.float64)

    for start in range(0, len(grid_pts), batch):
        end = min(start + batch, len(grid_pts))
        candidate_points = grid_pts[start:end]
        distances = np.linalg.norm(
            abs_pos[None, :, :] - candidate_points[:, None, :], axis=2
        )
        delays = distances / speed_of_sound
        pair_delays = delays[:, pair_i] - delays[:, pair_j]
        phase = np.exp(
            1j
            * pair_delays[:, :, None]
            * angular_frequencies[None, None, :]
        ).astype(np.complex64)
        scores[start:end] = np.real(
            np.sum(phase * cross_spectra[None, :, :], axis=(1, 2))
        )

    return scores


def covariance_matrices(
    transformed: np.ndarray,
    frequency_mask: np.ndarray,
    time_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = transformed[:, frequency_mask, :][:, :, time_mask]
    mic_count, frequency_count, active_count = selected.shape
    if active_count == 0:
        raise RuntimeError("No active frames available for covariance estimation.")

    covariance = np.empty((frequency_count, mic_count, mic_count), dtype=np.complex64)
    for frequency_index in range(frequency_count):
        x_frequency = selected[:, frequency_index, :]
        covariance[frequency_index] = (
            x_frequency @ np.conj(x_frequency.T)
        ) / active_count
    return selected, covariance


def das_score_from_tau_rel(
    transformed: np.ndarray,
    frequencies: np.ndarray,
    frequency_mask: np.ndarray,
    time_mask: np.ndarray,
    tau_rel_grid: np.ndarray,
    batch: int = 128,
) -> np.ndarray:
    """Frequency-domain Delay-and-Sum/Bartlett spectrum: a^H R a."""
    _, covariance = covariance_matrices(transformed, frequency_mask, time_mask)
    selected_frequencies = frequencies[frequency_mask].astype(np.float64)
    scores = np.zeros(len(tau_rel_grid), dtype=np.float64)

    for start in range(0, len(tau_rel_grid), batch):
        end = min(start + batch, len(tau_rel_grid))
        tau_batch = tau_rel_grid[start:end]
        batch_scores = np.zeros(end - start, dtype=np.float64)

        for frequency_index, frequency in enumerate(selected_frequencies):
            steering = np.exp(
                -1j * 2.0 * np.pi * frequency * tau_batch
            ).astype(np.complex64)
            # Correct Bartlett quadratic form for row-wise steering vectors.
            r_times_a = (covariance[frequency_index] @ steering.T).T
            batch_scores += np.real(
                np.sum(np.conj(steering) * r_times_a, axis=1)
            )

        scores[start:end] = batch_scores

    return scores


def mvdr_score_from_tau_rel(
    transformed: np.ndarray,
    frequencies: np.ndarray,
    frequency_mask: np.ndarray,
    time_mask: np.ndarray,
    tau_rel_grid: np.ndarray,
    diagonal_loading_factor: float = 1e-2,
    batch: int = 128,
) -> np.ndarray:
    """Wideband near-field MVDR/Capon spatial spectrum."""
    selected, _ = covariance_matrices(transformed, frequency_mask, time_mask)
    mic_count, frequency_count, active_count = selected.shape
    selected_frequencies = frequencies[frequency_mask].astype(np.float64)
    scores = np.zeros(len(tau_rel_grid), dtype=np.float64)
    identity = np.eye(mic_count, dtype=np.complex128)

    for frequency_index in range(frequency_count):
        x_frequency = selected[:, frequency_index, :]
        covariance = (x_frequency @ np.conj(x_frequency.T)) / active_count
        covariance = covariance.astype(np.complex128)
        loading = diagonal_loading_factor * (np.trace(covariance).real / mic_count)
        loaded = covariance + max(loading, EPS) * identity

        for start in range(0, len(tau_rel_grid), batch):
            end = min(start + batch, len(tau_rel_grid))
            tau_batch = tau_rel_grid[start:end]
            steering = np.exp(
                -1j * 2.0 * np.pi * selected_frequencies[frequency_index] * tau_batch
            ).astype(np.complex128)
            solved = np.linalg.solve(loaded, steering.T)
            denominator = np.sum(np.conj(steering) * solved.T, axis=1)
            scores[start:end] += 1.0 / np.maximum(np.real(denominator), EPS)

    return scores


# ---------------------------
# Plotting and result formatting
# ---------------------------

def save_slice_plot(
    scores: np.ndarray,
    grid_shape: tuple[int, int, int],
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    peaks: list[int],
    out_png: Path,
    title: str,
    truth: np.ndarray | None = None,
) -> None:
    ny, nx, nz = grid_shape
    score_volume = scores.reshape((ny, nx, nz))
    strongest_y, strongest_x, strongest_z = np.unravel_index(peaks[0], grid_shape)
    z_value = zs[strongest_z]
    plane = score_volume[:, :, strongest_z]

    plt.figure(figsize=(8, 4.5))
    plt.imshow(
        plane,
        origin="lower",
        aspect="auto",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
    )
    plt.colorbar(label="score")

    for number, peak in enumerate(peaks, start=1):
        iy, ix, iz = np.unravel_index(peak, grid_shape)
        plt.scatter(xs[ix], ys[iy], marker="x", label="estimate" if number == 1 else None)
        plt.text(xs[ix], ys[iy], f"E{number} z={zs[iz]:.2f}", fontsize=8)

    if truth is not None and truth.size:
        plt.scatter(truth[:, 0], truth[:, 1], marker="o", facecolors="none", label="truth")
        for number, position in enumerate(truth, start=1):
            plt.text(position[0], position[1], f"T{number} z={position[2]:.2f}", fontsize=8)

    plt.title(f"{title} (displayed z-slice = {z_value:.2f} m)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def add_method_results(
    row: dict[str, object],
    method: str,
    grid: np.ndarray,
    peaks: list[int],
    scores: np.ndarray,
    truth: np.ndarray | None,
) -> None:
    estimated = grid[peaks]
    row[f"{method}_peak_count"] = len(peaks)

    for index, (position, peak_index) in enumerate(zip(estimated, peaks, strict=True), start=1):
        row[f"{method}_raw_x{index}"] = float(position[0])
        row[f"{method}_raw_y{index}"] = float(position[1])
        row[f"{method}_raw_z{index}"] = float(position[2])
        row[f"{method}_raw_score{index}"] = float(scores[peak_index])

    if truth is None:
        return

    matched = match_estimates_to_truth(estimated, truth)
    row[f"{method}_mean_error"] = matched["mean_error"]
    row[f"{method}_median_error"] = matched["median_error"]
    row[f"{method}_max_error"] = matched["max_error"]

    matched_estimated = np.asarray(matched["estimated"])
    matched_truth = np.asarray(matched["truth"])
    estimate_indices = np.asarray(matched["estimate_indices"], dtype=int)
    truth_indices = np.asarray(matched["truth_indices"], dtype=int)
    errors = np.asarray(matched["errors_3d"])
    depth_errors = np.asarray(matched["depth_errors"])
    in_plane_errors = np.asarray(matched["in_plane_errors"])

    for index in range(len(errors)):
        output_index = index + 1
        row[f"{method}_truth_index{output_index}"] = int(truth_indices[index]) + 1
        # CSV indices are one-based; estimate indices refer to the raw peak order.
        row[f"{method}_estimate_index{output_index}"] = int(estimate_indices[index]) + 1
        row[f"{method}_est_x{output_index}"] = float(matched_estimated[index, 0])
        row[f"{method}_est_y{output_index}"] = float(matched_estimated[index, 1])
        row[f"{method}_est_z{output_index}"] = float(matched_estimated[index, 2])
        row[f"{method}_truth_x{output_index}"] = float(matched_truth[index, 0])
        row[f"{method}_truth_y{output_index}"] = float(matched_truth[index, 1])
        row[f"{method}_truth_z{output_index}"] = float(matched_truth[index, 2])
        row[f"{method}_error{output_index}"] = float(errors[index])
        row[f"{method}_depth_error{output_index}"] = float(depth_errors[index])
        row[f"{method}_in_plane_error{output_index}"] = float(in_plane_errors[index])


def collect_wav_paths(single_wav: str | None, glob_pattern: str | None) -> list[Path]:
    if bool(single_wav) == bool(glob_pattern):
        raise ValueError("Provide exactly one of --wav or --glob.")
    if single_wav:
        path = Path(single_wav)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return [path]

    paths = sorted(Path(".").glob(str(glob_pattern)))
    if not paths:
        raise FileNotFoundError(f"No WAV files matched: {glob_pattern}")
    return paths

SOURCE_COLORS = [
    "#E41A1C",  # red
    "#CC00CC",  # magenta
    "#FF7F00",  # orange
]


def save_orthogonal_projections(
    scores: np.ndarray,
    grid_shape: tuple[int, int, int],
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    peaks: list[int],
    out_prefix: Path,
    title: str,
    truth: np.ndarray | None = None,
    normalize: bool = True,
) -> None:
    """Save XY, XZ and YZ maximum projections of a 3D score map.

    Truths are open circles and estimates are crosses. When ground truth is
    available, estimates are optimally matched to truths before colors and
    source numbers are assigned.
    """

    score_volume = scores.reshape(grid_shape).astype(np.float64)

    if normalize:
        score_min = float(np.min(score_volume))
        score_max = float(np.max(score_volume))

        if score_max > score_min:
            display_volume = (
                score_volume - score_min
            ) / (score_max - score_min)
        else:
            display_volume = np.zeros_like(score_volume)

        colorbar_label = "normalized score"
    else:
        display_volume = score_volume
        colorbar_label = "score"

    # Convert peak indices to physical coordinates.
    raw_estimates = []

    for peak in peaks:
        iy, ix, iz = np.unravel_index(peak, grid_shape)
        raw_estimates.append(
            [xs[ix], ys[iy], zs[iz]]
        )

    raw_estimates = np.asarray(
        raw_estimates,
        dtype=np.float64,
    ).reshape((-1, 3))

    # Reorder estimates according to the same optimal truth matching used
    # for the numerical evaluation.
    if truth is not None and truth.size:
        truth_array = np.asarray(
            truth,
            dtype=np.float64,
        ).reshape((-1, 3))

        matched = match_estimates_to_truth(
            raw_estimates,
            truth_array,
        )

        estimate_coordinates = np.asarray(
            matched["estimated"],
            dtype=np.float64,
        )

        truth_coordinates = np.asarray(
            matched["truth"],
            dtype=np.float64,
        )
    else:
        estimate_coordinates = raw_estimates
        truth_coordinates = np.empty(
            (0, 3),
            dtype=np.float64,
        )

    # score_volume dimensions are [y, x, z].
    projections = [
        {
            "suffix": "xy_max_z",
            "plane": np.max(display_volume, axis=2),
            "horizontal_values": xs,
            "vertical_values": ys,
            "horizontal_coordinate": 0,
            "vertical_coordinate": 1,
            "horizontal_label": "x (m)",
            "vertical_label": "y (m)",
            "description": "maximum over z",
        },
        {
            "suffix": "xz_max_y",
            "plane": np.max(display_volume, axis=0).T,
            "horizontal_values": xs,
            "vertical_values": zs,
            "horizontal_coordinate": 0,
            "vertical_coordinate": 2,
            "horizontal_label": "x (m)",
            "vertical_label": "z (m)",
            "description": "maximum over y",
        },
        {
            "suffix": "yz_max_x",
            "plane": np.max(display_volume, axis=1).T,
            "horizontal_values": ys,
            "vertical_values": zs,
            "horizontal_coordinate": 1,
            "vertical_coordinate": 2,
            "horizontal_label": "y (m)",
            "vertical_label": "z (m)",
            "description": "maximum over x",
        },
    ]

    for projection in projections:
        figure, axis = plt.subplots(figsize=(7.5, 5.5))

        image = axis.imshow(
            projection["plane"],
            origin="lower",
            aspect="auto",
            extent=[
                projection["horizontal_values"].min(),
                projection["horizontal_values"].max(),
                projection["vertical_values"].min(),
                projection["vertical_values"].max(),
            ],
            vmin=0.0 if normalize else None,
            vmax=1.0 if normalize else None,
        )

        figure.colorbar(
            image,
            ax=axis,
            label=colorbar_label,
        )

        horizontal_index = projection["horizontal_coordinate"]
        vertical_index = projection["vertical_coordinate"]

        # Draw each matched truth/estimate pair using the same color.
        for index, estimate_position in enumerate(estimate_coordinates):
            color = SOURCE_COLORS[index % len(SOURCE_COLORS)]

            if len(truth_coordinates):
                truth_position = truth_coordinates[index]

                axis.scatter(
                    truth_position[horizontal_index],
                    truth_position[vertical_index],
                    marker="o",
                    s=120,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=2.2,
                    label=f"truth {index + 1}",
                    zorder=4,
                )

            axis.scatter(
                estimate_position[horizontal_index],
                estimate_position[vertical_index],
                marker="x",
                s=120,
                c=color,
                linewidths=2.4,
                label=f"estimate {index + 1}",
                zorder=5,
            )

        axis.set_xlabel(projection["horizontal_label"])
        axis.set_ylabel(projection["vertical_label"])
        axis.set_title(
            f"{title}: {projection['description']}"
        )
        axis.legend(
            loc="best",
            fontsize=8,
        )

        figure.tight_layout()

        output_path = (
            out_prefix.parent
            / f"{out_prefix.name}_{projection['suffix']}.png"
        )

        figure.savefig(
            output_path,
            dpi=200,
        )
        plt.close(figure)

def save_relevant_z_slices(
    scores: np.ndarray,
    grid_shape: tuple[int, int, int],
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    peaks: list[int],
    out_prefix: Path,
    title: str,
    truth: np.ndarray | None = None,
) -> None:
    """Save XY slices at truth z-levels and show all truths and estimates together.

    Truths are open circles.
    Estimates are crosses.
    Color identifies the truth/estimate pair after optimal matching.
    """

    score_volume = scores.reshape(grid_shape).astype(np.float64)

    score_min = float(np.min(score_volume))
    score_max = float(np.max(score_volume))

    if score_max > score_min:
        display_volume = (score_volume - score_min) / (score_max - score_min)
    else:
        display_volume = np.zeros_like(score_volume)

    # Convert peak indices into physical coordinates.
    raw_estimates = []
    for peak in peaks:
        iy, ix, iz = np.unravel_index(peak, grid_shape)
        raw_estimates.append([xs[ix], ys[iy], zs[iz]])

    raw_estimates = np.asarray(raw_estimates, dtype=np.float64)

    if truth is not None and truth.size:
        truth_positions = np.asarray(truth, dtype=np.float64)

        # Reorder estimates so estimate i corresponds to truth i.
        matched = match_estimates_to_truth(raw_estimates, truth_positions)
        estimate_positions = np.asarray(matched["estimated"], dtype=np.float64)
        truth_positions = np.asarray(matched["truth"], dtype=np.float64)

        # Save only slices corresponding to truth heights.
        relevant_z_indices = sorted(
            {
                int(np.argmin(np.abs(zs - position[2])))
                for position in truth_positions
            }
        )
    else:
        truth_positions = np.empty((0, 3), dtype=np.float64)
        estimate_positions = raw_estimates

        # Without truth, use estimated source heights.
        relevant_z_indices = sorted(
            {
                int(np.argmin(np.abs(zs - position[2])))
                for position in estimate_positions
            }
        )

    for iz in relevant_z_indices:
        current_z = float(zs[iz])

        plt.figure(figsize=(7.5, 5.5))

        image = plt.imshow(
            display_volume[:, :, iz],
            origin="lower",
            aspect="auto",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            vmin=0.0,
            vmax=1.0,
        )
        plt.colorbar(image, label="normalized score")

        # Plot matched truth/estimate pairs with the same color.
        for index, estimate in enumerate(estimate_positions):
            color = SOURCE_COLORS[index % len(SOURCE_COLORS)]

            estimate_on_slice = np.isclose(
                estimate[2],
                current_z,
                atol=(zs[1] - zs[0]) / 2 if len(zs) > 1 else 1e-12,
            )

            plt.scatter(
                estimate[0],
                estimate[1],
                marker="x",
                s=120,
                c=color,
                linewidths=2.4 if estimate_on_slice else 1.7,
                alpha=1.0 if estimate_on_slice else 0.4,
                label=f"Estimate {index + 1}",
                zorder=4,
            )

        for index, position in enumerate(truth_positions):
            color = SOURCE_COLORS[index % len(SOURCE_COLORS)]

            truth_on_slice = np.isclose(
                position[2],
                current_z,
                atol=(zs[1] - zs[0]) / 2 if len(zs) > 1 else 1e-12,
            )

            plt.scatter(
                position[0],
                position[1],
                marker="o",
                s=130,
                facecolors="none",
                edgecolors=color,
                linewidths=2.4 if truth_on_slice else 1.7,
                alpha=1.0 if truth_on_slice else 0.4,
                label=f"Truth {index + 1}",
                zorder=5,
            )

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"{title}: z-slice at {current_z:.2f} m")

        # Every saved slice receives a complete legend.
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_entries = dict(zip(labels, handles))
        plt.legend(
            unique_entries.values(),
            unique_entries.keys(),
            loc="best",
            fontsize=8,
        )

        plt.tight_layout()

        output_path = (
            out_prefix.parent
            / f"{out_prefix.name}_z_{current_z:.2f}.png"
        )
        plt.savefig(output_path, dpi=200)
        plt.close()



# ---------------------------
# Main
# ---------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Near-field localization of simulated bird recordings."
    )
    parser.add_argument("--csv", default="microphone_positions.csv")
    parser.add_argument("--wav", default=None, help="Process one WAV file.")
    parser.add_argument("--glob", default=None, help='Process multiple WAVs, e.g. "data/01*.wav".')
    parser.add_argument(
        "--txt",
        default=None,
        help="Metadata file for a single WAV; otherwise WAV stem + .txt is used.",
    )
    parser.add_argument("--outdir", default="out")
    parser.add_argument("--num_sources", type=int, choices=[1, 3], required=True)

    parser.add_argument("--mics_use", type=int, default=32)
    parser.add_argument(
        "--mic_selection",
        choices=["farthest", "id"],
        default="farthest",
    )
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--noverlap", type=int, default=768)
    parser.add_argument("--fmin", type=float, default=1000.0)
    parser.add_argument("--fmax", type=float, default=16000.0)
    parser.add_argument("--active_quantile", type=float, default=0.70)

    parser.add_argument("--xmin", type=float, default=2.0)
    parser.add_argument("--xmax", type=float, default=5.0)
    parser.add_argument("--ymin", type=float, default=0.2)
    parser.add_argument("--ymax", type=float, default=3.6)
    parser.add_argument("--zmin", type=float, default=0.0)
    parser.add_argument("--zmax", type=float, default=2.8)
    parser.add_argument("--step", type=float, default=0.1)

    parser.add_argument(
        "--min_sep",
        type=float,
        default=0.2,
        help="Minimum distance in meters between selected source estimates.",
    )
    parser.add_argument("--speed_of_sound", type=float, default=343.0)
    parser.add_argument("--mvdr_dl", type=float, default=1e-2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--skip_mvdr", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.txt is not None and args.glob is not None:
        parser.error("--txt cannot be used together with --glob.")

    center, all_mic_positions = load_mic_positions(args.csv)
    print("Array centre:", center)
    print("Available microphones:", len(all_mic_positions))

    mic_indices = choose_mics(all_mic_positions, args.mics_use, args.mic_selection)
    mic_positions = all_mic_positions[mic_indices]

    output_directory = Path(args.outdir)
    output_directory.mkdir(parents=True, exist_ok=True)
    wav_paths = collect_wav_paths(args.wav, args.glob)
    result_rows: list[dict[str, object]] = []

    for wav_path in wav_paths:
        metadata_path = Path(args.txt) if args.txt else wav_path.with_suffix(".txt")
        truth = parse_truth_xyz(metadata_path) if metadata_path.exists() else None

        print(f"\n--- Processing {wav_path.name} ---")
        if truth is None:
            print("Ground truth: not available")
        else:
            print(f"Ground-truth sources: {len(truth)}")
            if len(truth) != args.num_sources:
                raise ValueError(
                    f"--num_sources={args.num_sources}, but {metadata_path.name} "
                    f"contains {len(truth)} source positions."
                )

        audio, sample_rate = sf.read(str(wav_path), always_2d=True)
        if audio.shape[1] != len(all_mic_positions):
            raise RuntimeError(
                f"Audio has {audio.shape[1]} channels, but the CSV contains "
                f"{len(all_mic_positions)} microphones."
            )

        selected_audio = audio[:, mic_indices]
        print(f"Audio: {audio.shape[0]} samples, {sample_rate} Hz")
        print(f"Using {len(mic_indices)} microphones ({args.mic_selection} selection)")

        frequencies, times, transformed = stft_multich(
            selected_audio,
            sample_rate,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
        )
        frequency_mask = (frequencies >= args.fmin) & (frequencies <= args.fmax)
        time_mask = activity_mask(transformed, frequency_mask, args.active_quantile)
        print(f"Active frames: {np.sum(time_mask)}/{len(times)}")

        grid, xs, ys, zs, grid_shape = make_grid(
            (args.xmin, args.xmax),
            (args.ymin, args.ymax),
            (args.zmin, args.zmax),
            args.step,
        )
        print(f"Grid points: {len(grid)}; shape={grid_shape}; step={args.step} m")

        row: dict[str, object] = {
            "file": wav_path.name,
            "sample_rate": int(sample_rate),
            "channels": int(audio.shape[1]),
            "mics_used": int(len(mic_indices)),
            "mic_selection": args.mic_selection,
            "num_sources": args.num_sources,
            "grid_step": args.step,
            "min_sep": args.min_sep,
            "fmin": args.fmin,
            "fmax": args.fmax,
            "active_quantile": args.active_quantile,
        }

        if truth is not None:
            oracle_errors = nearest_grid_errors(grid, truth)
            row["oracle_grid_mean_error"] = float(np.mean(oracle_errors))
            row["oracle_grid_max_error"] = float(np.max(oracle_errors))
            for index, error in enumerate(oracle_errors, start=1):
                row[f"oracle_grid_error{index}"] = float(error)

        print("Computing SRP-PHAT...")
        srp_scores = srp_phat_score(
            mic_positions,
            transformed,
            frequencies,
            frequency_mask,
            time_mask,
            grid,
            speed_of_sound=args.speed_of_sound,
            batch=args.batch,
        )
        srp_peaks = nms_top_k_peaks(
            grid, srp_scores, grid_shape, args.num_sources, args.min_sep
        )
        add_method_results(row, "srp", grid, srp_peaks, srp_scores, truth)
   
        save_orthogonal_projections(
            srp_scores,
            grid_shape,
            xs,
            ys,
            zs,
            srp_peaks,
            output_directory / f"{wav_path.stem}_srp",
            "SRP-PHAT",
            truth,
        )

        save_relevant_z_slices(
            srp_scores,
            grid_shape,
            xs,
            ys,
            zs,
            srp_peaks,
            output_directory / f"{wav_path.stem}_srp",
            "SRP-PHAT",
            truth,
        )

        tau_rel_grid = compute_tau_rel_grid(
            mic_positions,
            grid,
            speed_of_sound=args.speed_of_sound,
        )

        print("Computing DAS/Bartlett...")
        das_scores = das_score_from_tau_rel(
            transformed,
            frequencies,
            frequency_mask,
            time_mask,
            tau_rel_grid,
            batch=args.batch,
        )
        das_peaks = nms_top_k_peaks(
            grid, das_scores, grid_shape, args.num_sources, args.min_sep
        )
        add_method_results(row, "das", grid, das_peaks, das_scores, truth)
        save_orthogonal_projections(
            das_scores,
            grid_shape,
            xs,
            ys,
            zs,
            das_peaks,
            output_directory / f"{wav_path.stem}_das",
            "DAS/Bartlett",
            truth,
        )

        save_relevant_z_slices(
            das_scores,
            grid_shape,
            xs,
            ys,
            zs,
            das_peaks,
            output_directory / f"{wav_path.stem}_das",
            "DAS/Bartlett",
            truth,
        )

        if not args.skip_mvdr:
            print("Computing MVDR/Capon...")
            mvdr_scores = mvdr_score_from_tau_rel(
                transformed,
                frequencies,
                frequency_mask,
                time_mask,
                tau_rel_grid,
                diagonal_loading_factor=args.mvdr_dl,
                batch=args.batch,
            )
            mvdr_peaks = nms_top_k_peaks(
                grid, mvdr_scores, grid_shape, args.num_sources, args.min_sep
            )
            add_method_results(row, "mvdr", grid, mvdr_peaks, mvdr_scores, truth)
            save_orthogonal_projections(
                mvdr_scores,
                grid_shape,
                xs,
                ys,
                zs,
                mvdr_peaks,
                output_directory / f"{wav_path.stem}_mvdr",
                "MVDR/Capon",
                truth,
            )

            save_relevant_z_slices(
                mvdr_scores,
                grid_shape,
                xs,
                ys,
                zs,
                mvdr_peaks,
                output_directory / f"{wav_path.stem}_mvdr",
                "MVDR/Capon",
                truth,
            )

        result_rows.append(row)
        pd.DataFrame(result_rows).to_csv(output_directory / "results.csv", index=False)
        print("Saved intermediate results.csv")

    print(f"\nCompleted {len(result_rows)} recording(s).")
    print("Results:", output_directory / "results.csv")


if __name__ == "__main__":
    main()
