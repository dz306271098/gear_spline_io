#!/usr/bin/env python3
"""
Trajectory evaluation script for gear_spline_lio
Computes ATE (Absolute Trajectory Error) and RPE (Relative Pose Error)

Usage:
    python3 evaluate_trajectory.py <estimated_traj.txt> <ground_truth.txt>

Both files should be in TUM format:
    timestamp tx ty tz qx qy qz qw
"""

import numpy as np
import sys


def load_tum_trajectory(filename):
    """Load trajectory from TUM format file."""
    data = np.loadtxt(filename)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]  # qx, qy, qz, qw
    return timestamps, positions, quaternions


def associate_trajectories(ts_est, ts_gt, max_diff=0.05):
    """Associate estimated and ground truth trajectories by timestamp."""
    associations = []
    for i, t_est in enumerate(ts_est):
        # Find closest ground truth timestamp
        diffs = np.abs(ts_gt - t_est)
        j = np.argmin(diffs)
        if diffs[j] < max_diff:
            associations.append((i, j))
    return associations


def compute_ate(pos_est, pos_gt, associations):
    """Compute Absolute Trajectory Error (ATE)."""
    if len(associations) == 0:
        return float('inf'), float('inf'), float('inf')

    errors = []
    for i, j in associations:
        error = np.linalg.norm(pos_est[i] - pos_gt[j])
        errors.append(error)

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    return rmse, mean_error, max_error


def compute_rpe(pos_est, pos_gt, associations, delta=1):
    """Compute Relative Pose Error (RPE)."""
    if len(associations) < delta + 1:
        return float('inf'), float('inf'), float('inf')

    errors = []
    for k in range(len(associations) - delta):
        i1, j1 = associations[k]
        i2, j2 = associations[k + delta]

        # Relative displacement in estimated trajectory
        delta_est = pos_est[i2] - pos_est[i1]
        # Relative displacement in ground truth
        delta_gt = pos_gt[j2] - pos_gt[j1]

        # Error is the difference in relative displacements
        error = np.linalg.norm(delta_est - delta_gt)
        errors.append(error)

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    return rmse, mean_error, max_error


def align_trajectories(pos_est, pos_gt, associations):
    """
    Align estimated trajectory to ground truth using SVD (Horn's method).
    Returns: rotation, translation, scale
    """
    if len(associations) < 3:
        return np.eye(3), np.zeros(3), 1.0

    # Extract matched points
    pts_est = np.array([pos_est[i] for i, j in associations])
    pts_gt = np.array([pos_gt[j] for i, j in associations])

    # Center the points
    centroid_est = np.mean(pts_est, axis=0)
    centroid_gt = np.mean(pts_gt, axis=0)

    pts_est_centered = pts_est - centroid_est
    pts_gt_centered = pts_gt - centroid_gt

    # Compute scale
    scale_est = np.sqrt(np.sum(pts_est_centered**2))
    scale_gt = np.sqrt(np.sum(pts_gt_centered**2))
    scale = scale_gt / scale_est if scale_est > 1e-10 else 1.0

    # Compute rotation using SVD
    H = pts_est_centered.T @ pts_gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_gt - scale * R @ centroid_est

    return R, t, scale


def transform_trajectory(positions, R, t, scale):
    """Apply transformation to trajectory."""
    return scale * (R @ positions.T).T + t


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 evaluate_trajectory.py <estimated_traj.txt> <ground_truth.txt>")
        print("       Optional: --align  to align trajectories before evaluation")
        sys.exit(1)

    est_file = sys.argv[1]
    gt_file = sys.argv[2]
    do_align = "--align" in sys.argv

    print("=" * 60)
    print("Trajectory Evaluation")
    print("=" * 60)
    print(f"Estimated: {est_file}")
    print(f"Ground Truth: {gt_file}")
    print(f"Alignment: {'Yes' if do_align else 'No'}")
    print()

    # Load trajectories
    try:
        ts_est, pos_est, quat_est = load_tum_trajectory(est_file)
        ts_gt, pos_gt, quat_gt = load_tum_trajectory(gt_file)
    except Exception as e:
        print(f"Error loading trajectories: {e}")
        sys.exit(1)

    print(f"Estimated trajectory: {len(ts_est)} poses")
    print(f"Ground truth: {len(ts_gt)} poses")
    print(f"Time range (est): [{ts_est[0]:.2f}, {ts_est[-1]:.2f}]")
    print(f"Time range (gt):  [{ts_gt[0]:.2f}, {ts_gt[-1]:.2f}]")
    print()

    # Associate trajectories
    associations = associate_trajectories(ts_est, ts_gt)
    print(f"Associated poses: {len(associations)}")

    if len(associations) == 0:
        print("ERROR: No poses could be associated!")
        print("Check that timestamps overlap between estimated and ground truth.")
        sys.exit(1)

    # Align if requested
    if do_align:
        R, t, scale = align_trajectories(pos_est, pos_gt, associations)
        pos_est_aligned = transform_trajectory(pos_est, R, t, scale)
        print(f"Alignment scale: {scale:.6f}")
        print(f"Alignment translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    else:
        pos_est_aligned = pos_est

    print()

    # Compute ATE
    ate_rmse, ate_mean, ate_max = compute_ate(pos_est_aligned, pos_gt, associations)
    print("ATE (Absolute Trajectory Error):")
    print(f"  RMSE: {ate_rmse:.4f} m")
    print(f"  Mean: {ate_mean:.4f} m")
    print(f"  Max:  {ate_max:.4f} m")
    print()

    # Compute RPE at different intervals
    for delta in [1, 10, 50]:
        rpe_rmse, rpe_mean, rpe_max = compute_rpe(pos_est_aligned, pos_gt, associations, delta)
        print(f"RPE (delta={delta}):")
        print(f"  RMSE: {rpe_rmse:.4f} m")
        print(f"  Mean: {rpe_mean:.4f} m")
        print(f"  Max:  {rpe_max:.4f} m")
        print()

    # Quality assessment
    print("=" * 60)
    print("Quality Assessment:")
    if ate_rmse < 0.03:
        print("  ATE: EXCELLENT (< 3cm)")
    elif ate_rmse < 0.05:
        print("  ATE: GOOD (< 5cm)")
    elif ate_rmse < 0.10:
        print("  ATE: ACCEPTABLE (< 10cm)")
    else:
        print(f"  ATE: NEEDS IMPROVEMENT (>= 10cm, actual: {ate_rmse:.2f}m)")
    print("=" * 60)


if __name__ == "__main__":
    main()
