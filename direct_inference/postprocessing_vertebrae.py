#!/usr/bin/env python3
"""
Postprocess AI-predicted vertebrae masks to reduce artifacts, fragmentation,
label overlaps, and anatomical ordering errors.  See vertebrae.md step 4.

Pipeline
--------
1.  Load ``combined_labels.nii.gz``
2.  Auto-detect superior-inferior (SI) axis from the NIfTI affine
3.  Per-label connected-component cleanup (keep largest, drop tiny fragments)
4.  Spine-centerline outlier removal (discard blobs far from the main spine)
5.  Resolve label overlaps via Euclidean distance transforms
6.  Enforce anatomical ordering along the SI axis (handles partial scans)
7.  Morphological cleanup (hole filling, binary closing)
8.  Volume & adjacency sanity checks
9.  Save refined combined + per-label masks
"""
import argparse
import os
import sys

import numpy as np
import nibabel as nib
import cc3d
import fastremap
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

VERTEBRAE_LABELS = list(range(1, 25))
CLASS_MAP_VERTEBRAE = {
    1: "vertebrae_L5",  2: "vertebrae_L4",  3: "vertebrae_L3",
    4: "vertebrae_L2",  5: "vertebrae_L1",
    6: "vertebrae_T12", 7: "vertebrae_T11", 8: "vertebrae_T10",
    9: "vertebrae_T9",  10: "vertebrae_T8", 11: "vertebrae_T7",
    12: "vertebrae_T6", 13: "vertebrae_T5", 14: "vertebrae_T4",
    15: "vertebrae_T3", 16: "vertebrae_T2", 17: "vertebrae_T1",
    18: "vertebrae_C7", 19: "vertebrae_C6", 20: "vertebrae_C5",
    21: "vertebrae_C4", 22: "vertebrae_C3", 23: "vertebrae_C2",
    24: "vertebrae_C1",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_si_axis(affine):
    """Return ``(axis_index, increasing_is_superior)`` from a NIfTI affine.

    Uses ``nibabel.aff2axcodes`` to find which voxel axis maps to the
    superior-inferior direction.  Falls back to axis 2 / True when the
    affine is ambiguous.
    """
    codes = nib.aff2axcodes(affine)
    for i, code in enumerate(codes):
        if code == "S":
            return i, True
        if code == "I":
            return i, False
    return 2, True


def compute_centroid(mask):
    """Return ``(i, j, k)`` centroid of a binary mask, or ``None`` if empty."""
    if not np.any(mask):
        return None
    ijk = np.array(np.where(mask > 0))
    return (float(ijk[0].mean()), float(ijk[1].mean()), float(ijk[2].mean()))


# ---------------------------------------------------------------------------
# Step 3 -- Per-label connected-component cleanup
# ---------------------------------------------------------------------------

def keep_topk_largest_connected_object(npy_mask, k, area_least, out_mask, out_label):
    """Keep the top-*k* largest connected components above *area_least*.
    Fills *out_mask* in-place with *out_label*.
    """
    labels_out = cc3d.connected_components(npy_mask.astype(np.uint8), connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    for i in range(min(k, len(candidates))):
        if candidates[i][1] > area_least:
            out_mask[labels_out == int(candidates[i][0])] = out_label


def extract_topk_largest_candidates(npy_mask, organ_num, area_least=0):
    """Extract the top *organ_num* largest connected components."""
    out_mask = np.zeros(npy_mask.shape, np.uint8)
    keep_topk_largest_connected_object(npy_mask, organ_num, area_least, out_mask, 1)
    return out_mask


def per_label_cleanup(masks, min_voxels, keep_top_k=1):
    """For each vertebra label: remove small components, keep the largest."""
    shape = next(iter(masks.values())).shape
    cleaned = {}
    for label in VERTEBRAE_LABELS:
        m = masks.get(label)
        if m is None or not np.any(m):
            cleaned[label] = np.zeros(shape, dtype=np.uint8)
            continue
        cleaned[label] = extract_topk_largest_candidates(
            m.astype(np.uint8), organ_num=keep_top_k, area_least=min_voxels,
        )
    return cleaned


# ---------------------------------------------------------------------------
# Step 4 -- Spine-centerline outlier removal
# ---------------------------------------------------------------------------

def remove_spine_outliers(masks, si_axis, max_deviation_voxels=60):
    """Discard vertebrae whose centroids lie far from the main spine line.

    Projects centroids onto the two axes orthogonal to *si_axis* and flags
    any whose distance from the robust median exceeds *max_deviation_voxels*.
    """
    other_axes = [i for i in range(3) if i != si_axis]
    centroids = {}
    for label in VERTEBRAE_LABELS:
        m = masks.get(label)
        if m is not None and np.any(m):
            c = compute_centroid(m)
            if c is not None:
                centroids[label] = c
    if len(centroids) < 3:
        return masks

    pos = np.array([[centroids[l][ax] for ax in other_axes] for l in centroids])
    median_pos = np.median(pos, axis=0)

    shape = next(iter(masks.values())).shape
    for label, c in list(centroids.items()):
        p = np.array([c[ax] for ax in other_axes])
        if np.linalg.norm(p - median_pos) > max_deviation_voxels:
            name = CLASS_MAP_VERTEBRAE.get(label, str(label))
            print(f"    Removed outlier: {name} (label {label})")
            masks[label] = np.zeros(shape, dtype=np.uint8)
    return masks


# ---------------------------------------------------------------------------
# Step 5 -- Overlap resolution via distance transforms
# ---------------------------------------------------------------------------

def resolve_overlaps(masks, shape):
    """Assign contested voxels to the label they are deepest inside of.

    For each label a Euclidean distance transform (EDT) measures how far each
    interior voxel is from the label boundary.  Where multiple labels claim
    the same voxel the one with the largest interior distance wins, giving a
    smooth Voronoi-like partition between adjacent vertebrae.
    """
    label_count = np.zeros(shape, dtype=np.int32)
    present_labels = []
    for label in VERTEBRAE_LABELS:
        m = masks.get(label)
        if m is not None and np.any(m):
            label_count += (m > 0).astype(np.int32)
            present_labels.append(label)

    combined = np.zeros(shape, dtype=np.int32)
    for label in present_labels:
        combined[masks[label] > 0] = label

    contested = label_count > 1
    if not np.any(contested):
        return combined

    ijk = np.array(np.where(contested))
    best_dist = np.full(ijk.shape[1], -1.0)
    best_label = np.zeros(ijk.shape[1], dtype=np.int32)

    for label in present_labels:
        m = masks[label]
        has_voxel = m[ijk[0], ijk[1], ijk[2]] > 0
        if not np.any(has_voxel):
            continue
        dt = distance_transform_edt(m > 0)
        d = dt[ijk[0], ijk[1], ijk[2]]
        better = (d > best_dist) & has_voxel
        best_dist[better] = d[better]
        best_label[better] = label

    combined[ijk[0], ijk[1], ijk[2]] = best_label
    return combined


# ---------------------------------------------------------------------------
# Step 6 -- Anatomical ordering enforcement (handles partial scans)
# ---------------------------------------------------------------------------

def enforce_anatomical_ordering(masks, si_axis, si_increasing_is_superior):
    """Re-assign labels so spatial order along SI matches label order.

    Label 1 (L5) is the most inferior; label 24 (C1) is the most superior.
    The *set* of label values present in the prediction is preserved -- only
    their spatial assignment is corrected.  This handles partial-spine scans
    correctly (e.g. only T1-L5 visible keeps labels 1-17 instead of wrongly
    remapping them to C1-T5).
    """
    centroids = {}
    for label in VERTEBRAE_LABELS:
        m = masks.get(label)
        if m is not None and np.any(m):
            c = compute_centroid(m)
            if c is not None:
                centroids[label] = c

    if len(centroids) < 2:
        return masks

    # Sort present blobs by SI position (inferior -> superior).
    # sign=+1 when increasing index = more superior (sort ascending gives
    # inferior first); sign=-1 when increasing index = more inferior.
    sign = 1 if si_increasing_is_superior else -1
    blobs_inferior_to_superior = sorted(
        centroids.keys(),
        key=lambda L: sign * centroids[L][si_axis],
    )

    sorted_label_values = sorted(centroids.keys())

    if blobs_inferior_to_superior == sorted_label_values:
        return masks

    new_for_old = {}
    for k, old_label in enumerate(blobs_inferior_to_superior):
        new_for_old[old_label] = sorted_label_values[k]

    shape = next(iter(masks.values())).shape
    refined = {label: np.zeros(shape, dtype=np.uint8) for label in VERTEBRAE_LABELS}
    for old_label, new_label in new_for_old.items():
        m = masks.get(old_label)
        if m is not None and np.any(m):
            refined[new_label][m > 0] = 1

    swaps = sum(1 for o, n in new_for_old.items() if o != n)
    if swaps:
        print(f"    Reordered {swaps} label(s) to fix spatial ordering")
    return refined


# ---------------------------------------------------------------------------
# Step 7 -- Morphological cleanup
# ---------------------------------------------------------------------------

def morphological_cleanup(masks, fill_holes=True, closing_size=3):
    """Fill holes and optionally apply binary closing per label."""
    shape = next(iter(masks.values())).shape
    structure = np.ones((closing_size,) * 3) if closing_size > 0 else None
    out = {}
    for label in VERTEBRAE_LABELS:
        m = masks.get(label)
        if m is None or not np.any(m):
            out[label] = np.zeros(shape, dtype=np.uint8)
            continue
        m = m.astype(bool)
        if fill_holes:
            m = ndimage.binary_fill_holes(m)
        if structure is not None:
            m = ndimage.binary_closing(m, structure=structure)
        out[label] = m.astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Step 8 -- Sanity checks
# ---------------------------------------------------------------------------

def volume_sanity_check(masks, case_id):
    """Print warnings for vertebrae with anomalous volumes."""
    volumes = []
    for label in VERTEBRAE_LABELS:
        m = masks.get(label)
        if m is not None and np.any(m):
            volumes.append((label, int(np.sum(m > 0))))
    if len(volumes) < 2:
        return
    vol_only = [v for _, v in volumes]
    median_vol = float(np.median(vol_only))
    for label, vol in volumes:
        if vol > 3.0 * median_vol or vol < 0.3 * median_vol:
            name = CLASS_MAP_VERTEBRAE.get(label, str(label))
            print(f"  [WARNING] {case_id} {name} (label {label}): "
                  f"volume={vol} voxels (median={median_vol:.0f})")


def adjacency_check(masks, si_axis, case_id):
    """Warn when consecutive present labels have a suspiciously large gap."""
    centroids = {}
    for label in VERTEBRAE_LABELS:
        m = masks.get(label)
        if m is not None and np.any(m):
            c = compute_centroid(m)
            if c is not None:
                centroids[label] = c[si_axis]
    present = sorted(centroids.keys())
    if len(present) < 2:
        return
    gaps = []
    for a, b in zip(present[:-1], present[1:]):
        gaps.append(abs(centroids[b] - centroids[a]))
    if not gaps:
        return
    median_gap = float(np.median(gaps))
    if median_gap == 0:
        return
    for i, (a, b) in enumerate(zip(present[:-1], present[1:])):
        if gaps[i] > 3.0 * median_gap:
            na = CLASS_MAP_VERTEBRAE.get(a, str(a))
            nb = CLASS_MAP_VERTEBRAE.get(b, str(b))
            print(f"  [WARNING] {case_id} large gap between {na} and {nb} "
                  f"({gaps[i]:.0f} vs median {median_gap:.0f} voxels)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_case(case_dir, output_dir, min_component_voxels, fill_holes,
                 closing_size, max_outlier_deviation):
    """Full postprocessing pipeline for one case."""
    combined_path = os.path.join(case_dir, "combined_labels.nii.gz")
    if not os.path.isfile(combined_path):
        print(f"  Skip (no combined_labels.nii.gz): {case_dir}")
        return
    case_id = os.path.basename(case_dir.rstrip("/"))

    print(f"  [{case_id}] Loading...")
    nii = nib.load(combined_path)
    data = np.asarray(nii.dataobj).astype(np.int32)
    shape = data.shape
    affine = nii.affine.copy()

    si_axis, si_up = detect_si_axis(affine)
    ax_names = {0: "i (dim 0)", 1: "j (dim 1)", 2: "k (dim 2)"}
    print(f"  [{case_id}] SI axis: {ax_names[si_axis]}, "
          f"increasing={'superior' if si_up else 'inferior'}")

    masks = {}
    for label in VERTEBRAE_LABELS:
        masks[label] = (data == label).astype(np.uint8)
    present = [l for l in VERTEBRAE_LABELS if np.any(masks[l])]
    print(f"  [{case_id}] Found {len(present)} vertebrae labels")

    print(f"  [{case_id}] Per-label cleanup...")
    masks = per_label_cleanup(masks, min_component_voxels, keep_top_k=1)

    print(f"  [{case_id}] Spine-centerline outlier removal...")
    masks = remove_spine_outliers(masks, si_axis, max_outlier_deviation)

    print(f"  [{case_id}] Resolving overlaps (distance transform)...")
    combined = resolve_overlaps(masks, shape)
    for label in VERTEBRAE_LABELS:
        masks[label] = (combined == label).astype(np.uint8)

    print(f"  [{case_id}] Enforcing anatomical ordering...")
    masks = enforce_anatomical_ordering(masks, si_axis, si_up)

    print(f"  [{case_id}] Morphological cleanup...")
    masks = morphological_cleanup(masks, fill_holes=fill_holes,
                                  closing_size=closing_size)

    print(f"  [{case_id}] Sanity checks...")
    volume_sanity_check(masks, case_id)
    adjacency_check(masks, si_axis, case_id)

    # ---- Save ----
    print(f"  [{case_id}] Saving...")
    out_combined = np.zeros(shape, dtype=np.uint8)
    for label in VERTEBRAE_LABELS:
        out_combined[masks[label] > 0] = label

    out_case_dir = os.path.join(output_dir, case_id)
    os.makedirs(out_case_dir, exist_ok=True)
    nib.save(
        nib.Nifti1Image(out_combined, affine),
        os.path.join(out_case_dir, "combined_labels.nii.gz"),
    )
    seg_dir = os.path.join(out_case_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    for label in VERTEBRAE_LABELS:
        name = CLASS_MAP_VERTEBRAE[label]
        nib.save(
            nib.Nifti1Image(masks[label].astype(np.uint8), affine),
            os.path.join(seg_dir, f"{name}.nii.gz"),
        )
    print(f"  [{case_id}] Done -> {out_case_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess vertebrae masks from SuPreM inference.",
    )
    parser.add_argument(
        "--input_dir", type=str, default="./AbdomenAtlasDemoPredict",
        help="Directory containing case folders with combined_labels.nii.gz",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./AbdomenAtlasDemoPredict_refined",
        help="Output directory for refined masks",
    )
    parser.add_argument(
        "--min_component_voxels", type=int, default=100,
        help="Minimum voxels to keep a connected component (smaller are removed)",
    )
    parser.add_argument(
        "--no_fill_holes", action="store_true",
        help="Disable binary hole filling",
    )
    parser.add_argument(
        "--closing_size", type=int, default=3,
        help="Binary closing structure size (0 to disable)",
    )
    parser.add_argument(
        "--max_outlier_deviation", type=float, default=60,
        help="Max lateral deviation from spine centerline in voxels "
             "(vertebrae further away are removed)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: input_dir not found: {args.input_dir}")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    case_dirs = []
    for name in sorted(os.listdir(args.input_dir)):
        path = os.path.join(args.input_dir, name)
        if os.path.isdir(path) and os.path.isfile(
                os.path.join(path, "combined_labels.nii.gz")):
            case_dirs.append(path)

    if not case_dirs:
        print(f"No case directories with combined_labels.nii.gz found in {args.input_dir}")
        sys.exit(0)

    print(f"Postprocessing {len(case_dirs)} case(s): "
          f"{args.input_dir} -> {args.output_dir}")
    for case_dir in case_dirs:
        process_case(
            case_dir, args.output_dir,
            min_component_voxels=args.min_component_voxels,
            fill_holes=not args.no_fill_holes,
            closing_size=args.closing_size,
            max_outlier_deviation=args.max_outlier_deviation,
        )
    print("All done.")


if __name__ == "__main__":
    main()
