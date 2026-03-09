#!/usr/bin/env python3
"""
Postprocess AI-predicted vertebrae masks to reduce artifacts, fragmentation,
label overlaps, and anatomical ordering errors.  See vertebrae.md step 4.

Pipeline
--------
 1. Load ``combined_labels.nii.gz`` (and optionally ``ct.nii.gz``)
 2. Auto-detect superior-inferior (SI) axis from the NIfTI affine
 3. Per-label connected-component cleanup (keep largest, drop fragments)
 4. Spine-centerline outlier removal (discard blobs far from the spine)
 5. Resolve label overlaps via Euclidean distance transforms
 6. Enforce anatomical ordering along the SI axis (handles partial scans)
 7. Fill gaps within the spine envelope (nearest-label assignment)
 8. Interpolate missing vertebrae in the label sequence
 9. Per-vertebra adaptive morphological regularization
10. Gaussian soft-voting label smoothing for clean boundaries
11. Final overlap cleanup after smoothing
12. (Optional) CT-guided bone-mask boundary refinement
13. Volume & adjacency sanity checks
14. Save refined combined + per-label masks
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


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def detect_si_axis(affine):
    """Return ``(axis_index, increasing_is_superior)`` from a NIfTI affine."""
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


def _present_labels(masks):
    """Return sorted list of labels that have non-empty masks."""
    return sorted(l for l in VERTEBRAE_LABELS
                  if masks.get(l) is not None and np.any(masks[l]))


def _masks_to_combined(masks, shape):
    """Merge per-label masks into a single label volume."""
    combined = np.zeros(shape, dtype=np.int32)
    for label in VERTEBRAE_LABELS:
        m = masks.get(label)
        if m is not None and np.any(m):
            combined[m > 0] = label
    return combined


def _combined_to_masks(combined):
    """Split a combined label volume into per-label binary masks."""
    return {label: (combined == label).astype(np.uint8) for label in VERTEBRAE_LABELS}


# ───────────────────────────────────────────────────────────────────────────
# Step 3 — Per-label connected-component cleanup
# ───────────────────────────────────────────────────────────────────────────

def keep_topk_largest_connected_object(npy_mask, k, area_least, out_mask, out_label):
    labels_out = cc3d.connected_components(npy_mask.astype(np.uint8), connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    for i in range(min(k, len(candidates))):
        if candidates[i][1] > area_least:
            out_mask[labels_out == int(candidates[i][0])] = out_label


def extract_topk_largest_candidates(npy_mask, organ_num, area_least=0):
    out_mask = np.zeros(npy_mask.shape, np.uint8)
    keep_topk_largest_connected_object(npy_mask, organ_num, area_least, out_mask, 1)
    return out_mask


def per_label_cleanup(masks, min_voxels, keep_top_k=1):
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


# ───────────────────────────────────────────────────────────────────────────
# Step 4 — Spine-centerline outlier removal
# ───────────────────────────────────────────────────────────────────────────

def remove_spine_outliers(masks, si_axis, max_deviation_voxels=60):
    """Discard vertebrae whose centroids are far from the median spine line."""
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


# ───────────────────────────────────────────────────────────────────────────
# Step 5 — Overlap resolution via distance transforms
# ───────────────────────────────────────────────────────────────────────────

def resolve_overlaps(masks, shape):
    """Assign contested voxels to the label they are deepest inside of."""
    label_count = np.zeros(shape, dtype=np.int32)
    present = []
    for label in VERTEBRAE_LABELS:
        m = masks.get(label)
        if m is not None and np.any(m):
            label_count += (m > 0).astype(np.int32)
            present.append(label)

    combined = np.zeros(shape, dtype=np.int32)
    for label in present:
        combined[masks[label] > 0] = label

    contested = label_count > 1
    if not np.any(contested):
        return combined

    ijk = np.array(np.where(contested))
    best_dist = np.full(ijk.shape[1], -1.0)
    best_label = np.zeros(ijk.shape[1], dtype=np.int32)

    for label in present:
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


# ───────────────────────────────────────────────────────────────────────────
# Step 6 — Anatomical ordering enforcement (handles partial scans)
# ───────────────────────────────────────────────────────────────────────────

def enforce_anatomical_ordering(masks, si_axis, si_increasing_is_superior):
    """Re-assign labels so spatial order along SI matches label order.

    The *set* of present label values is preserved; only their spatial
    assignment is corrected.  Partial scans keep their original label range.
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


# ───────────────────────────────────────────────────────────────────────────
# Step 7 — Fill gaps inside the spine envelope  (NEW)
# ───────────────────────────────────────────────────────────────────────────

def fill_spine_gaps(masks, si_axis, shape, gap_closing_radius=10):
    """Fill unlabeled voxels that lie *inside* the spine envelope.

    1. Build a binary spine mask from all labels.
    2. Close it with an elongated kernel along the SI axis to bridge
       inter-vertebra gaps.
    3. Apply a small isotropic closing to seal lateral holes.
    4. Fill 3-D holes in the envelope.
    5. Assign every unlabeled voxel inside the envelope to the nearest
       vertebra label via a distance transform.
    """
    combined = _masks_to_combined(masks, shape)
    labeled_mask = combined > 0
    if not np.any(labeled_mask):
        return masks

    # Elongated closing kernel along the SI axis to bridge gaps between vertebrae
    si_kernel_shape = [3, 3, 3]
    si_kernel_shape[si_axis] = gap_closing_radius * 2 + 1
    si_kernel = np.ones(si_kernel_shape, dtype=bool)
    spine_envelope = ndimage.binary_closing(labeled_mask, structure=si_kernel)

    # Small isotropic closing to seal any remaining lateral holes
    spine_envelope = ndimage.binary_closing(spine_envelope,
                                            structure=np.ones((5, 5, 5), dtype=bool))
    spine_envelope = ndimage.binary_fill_holes(spine_envelope)

    gaps = spine_envelope & ~labeled_mask
    n_gap = int(np.sum(gaps))
    if n_gap == 0:
        return masks
    print(f"    Filling {n_gap} gap voxels inside spine envelope")

    # Nearest-label assignment for gap voxels
    gap_ijk = np.array(np.where(gaps))
    best_dist = np.full(gap_ijk.shape[1], np.inf)
    best_label = np.zeros(gap_ijk.shape[1], dtype=np.int32)

    for label in _present_labels(masks):
        dt = distance_transform_edt(~(masks[label] > 0))
        d = dt[gap_ijk[0], gap_ijk[1], gap_ijk[2]]
        closer = d < best_dist
        best_dist[closer] = d[closer]
        best_label[closer] = label

    new_combined = combined.copy()
    new_combined[gap_ijk[0], gap_ijk[1], gap_ijk[2]] = best_label
    return _combined_to_masks(new_combined)


# ───────────────────────────────────────────────────────────────────────────
# Step 8 — Interpolate missing vertebrae  (NEW)
# ───────────────────────────────────────────────────────────────────────────

def interpolate_missing_vertebrae(masks, si_axis, si_increasing_is_superior, shape):
    """If there are gaps in the label sequence, synthesise masks for them.

    For each missing label between two present neighbors:
      - Estimate the centroid by linear interpolation.
      - Create an approximate mask by dilating a seed at the estimated
        centroid, then crop it to a reasonable volume (median of neighbors).
    """
    present = _present_labels(masks)
    if len(present) < 2:
        return masks

    centroids = {}
    volumes = {}
    for label in present:
        c = compute_centroid(masks[label])
        if c is not None:
            centroids[label] = np.array(c)
            volumes[label] = int(np.sum(masks[label] > 0))

    lo, hi = min(present), max(present)
    missing = [l for l in range(lo, hi + 1) if l not in present]
    if not missing:
        return masks

    median_vol = float(np.median(list(volumes.values())))
    structure = ndimage.generate_binary_structure(3, 1)

    for label in missing:
        below = [l for l in present if l < label]
        above = [l for l in present if l > label]
        if not below or not above:
            continue
        lb = max(below)
        la = min(above)
        if lb not in centroids or la not in centroids:
            continue

        frac = (label - lb) / (la - lb)
        est_centroid = centroids[lb] * (1 - frac) + centroids[la] * frac
        est_centroid = np.round(est_centroid).astype(int)
        est_centroid = np.clip(est_centroid, 0, np.array(shape) - 1)

        target_vol = int(median_vol * 0.7)

        seed = np.zeros(shape, dtype=np.uint8)
        seed[est_centroid[0], est_centroid[1], est_centroid[2]] = 1
        grown = seed.copy()
        for _ in range(80):
            grown = ndimage.binary_dilation(grown, structure=structure).astype(np.uint8)
            if int(np.sum(grown)) >= target_vol:
                break

        # Trim to target volume by keeping voxels closest to centroid
        ijk = np.array(np.where(grown > 0))
        if ijk.shape[1] > target_vol:
            dists = np.sum((ijk.T - est_centroid) ** 2, axis=1)
            keep = np.argsort(dists)[:target_vol]
            trimmed = np.zeros(shape, dtype=np.uint8)
            trimmed[ijk[0, keep], ijk[1, keep], ijk[2, keep]] = 1
            grown = trimmed

        # Only keep voxels that don't overlap existing labels
        existing = _masks_to_combined(masks, shape)
        grown[existing > 0] = 0

        if np.any(grown):
            masks[label] = grown
            name = CLASS_MAP_VERTEBRAE.get(label, str(label))
            print(f"    Interpolated missing {name} (label {label}), "
                  f"{int(np.sum(grown))} voxels")

    return masks


# ───────────────────────────────────────────────────────────────────────────
# Step 9 — Per-vertebra adaptive morphological regularization  (NEW)
# ───────────────────────────────────────────────────────────────────────────

def adaptive_morphological_regularization(masks, fill_holes=True, base_closing=3):
    """Morphological cleanup with per-vertebra adaptive kernel size.

    Larger vertebrae (lumbar) get a bigger closing kernel; smaller ones
    (cervical) get a smaller one.  This produces more anatomically
    consistent shapes than a single fixed kernel.
    """
    shape = next(iter(masks.values())).shape
    present = _present_labels(masks)
    if not present:
        return masks

    vol_map = {}
    for label in present:
        vol_map[label] = int(np.sum(masks[label] > 0))
    median_vol = float(np.median(list(vol_map.values()))) if vol_map else 1.0

    out = {}
    for label in VERTEBRAE_LABELS:
        m = masks.get(label)
        if m is None or not np.any(m):
            out[label] = np.zeros(shape, dtype=np.uint8)
            continue

        vol_ratio = vol_map.get(label, median_vol) / max(median_vol, 1.0)
        closing_sz = max(base_closing, int(base_closing * min(vol_ratio, 2.0)))
        closing_sz = closing_sz if closing_sz % 2 == 1 else closing_sz + 1

        m = m.astype(bool)
        if fill_holes:
            m = ndimage.binary_fill_holes(m)
        struct = np.ones((closing_sz,) * 3)
        m = ndimage.binary_closing(m, structure=struct)
        # Small opening to remove thin spurs
        if closing_sz >= 3:
            m = ndimage.binary_opening(m, structure=np.ones((3, 3, 3)))
        out[label] = m.astype(np.uint8)
    return out


# ───────────────────────────────────────────────────────────────────────────
# Step 10 — Gaussian soft-voting label smoothing  (NEW)
# ───────────────────────────────────────────────────────────────────────────

def smooth_labels_gaussian(combined, sigma=1.0, min_confidence=0.08):
    """Smooth label boundaries by Gaussian-weighted soft voting.

    Each label's binary mask is convolved with a Gaussian.  At every voxel
    the label with the highest blurred response wins, provided it exceeds
    *min_confidence* (prevents labels from bleeding far into background).
    The result has much smoother inter-vertebra boundaries.
    """
    labels_present = np.unique(combined)
    labels_present = labels_present[labels_present > 0]
    if len(labels_present) == 0:
        return combined

    scores = np.zeros(combined.shape + (len(labels_present),), dtype=np.float32)
    for i, label in enumerate(labels_present):
        scores[..., i] = ndimage.gaussian_filter(
            (combined == label).astype(np.float32), sigma=sigma,
        )

    max_score = np.max(scores, axis=-1)
    best_idx = np.argmax(scores, axis=-1)
    result = np.zeros_like(combined)
    has_label = max_score > min_confidence
    result[has_label] = labels_present[best_idx[has_label]]
    return result


# ───────────────────────────────────────────────────────────────────────────
# Step 12 — CT-guided bone-mask boundary refinement  (NEW, optional)
# ───────────────────────────────────────────────────────────────────────────

def ct_guided_refinement(masks, ct_data, shape,
                         bone_low=150, bone_high=3000,
                         dilate_bone=2):
    """Refine vertebra masks using CT bone intensity.

    Voxels claimed by a vertebra label but clearly outside the bone
    Hounsfield-unit range are removed.  A small dilation of the bone mask
    accounts for partial-volume effects at boundaries.
    """
    bone_mask = (ct_data >= bone_low) & (ct_data <= bone_high)
    if dilate_bone > 0:
        bone_mask = ndimage.binary_dilation(
            bone_mask,
            structure=ndimage.generate_binary_structure(3, 1),
            iterations=dilate_bone,
        )

    trimmed_total = 0
    out = {}
    for label in VERTEBRAE_LABELS:
        m = masks.get(label)
        if m is None or not np.any(m):
            out[label] = np.zeros(shape, dtype=np.uint8)
            continue
        refined = (m > 0) & bone_mask
        trimmed_total += int(np.sum(m > 0)) - int(np.sum(refined))
        # Ensure we keep at least the core (largest CC in intersection)
        if np.any(refined):
            out[label] = extract_topk_largest_candidates(
                refined.astype(np.uint8), organ_num=1, area_least=50,
            )
        else:
            out[label] = m  # keep original if bone mask removes everything
    if trimmed_total > 0:
        print(f"    CT-guided refinement trimmed {trimmed_total} non-bone voxels")
    return out


# ───────────────────────────────────────────────────────────────────────────
# Sanity checks
# ───────────────────────────────────────────────────────────────────────────

def volume_sanity_check(masks, case_id):
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


# ───────────────────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────────────────

def process_case(case_dir, output_dir, args, ct_root_path=None):
    """Full postprocessing pipeline for one case."""
    combined_path = os.path.join(case_dir, "combined_labels.nii.gz")
    if not os.path.isfile(combined_path):
        print(f"  Skip (no combined_labels.nii.gz): {case_dir}")
        return
    case_id = os.path.basename(case_dir.rstrip("/"))

    # ── Load ──
    print(f"  [{case_id}] Loading prediction...")
    nii = nib.load(combined_path)
    data = np.asarray(nii.dataobj).astype(np.int32)
    shape = data.shape
    affine = nii.affine.copy()

    ct_data = None
    if ct_root_path:
        ct_path = os.path.join(ct_root_path, case_id, "ct.nii.gz")
        if os.path.isfile(ct_path):
            print(f"  [{case_id}] Loading CT for guided refinement...")
            ct_nii = nib.load(ct_path)
            ct_data = np.asarray(ct_nii.dataobj).astype(np.float32)
            if ct_data.shape != shape:
                print(f"  [{case_id}] CT shape mismatch, skipping CT guidance")
                ct_data = None

    si_axis, si_up = detect_si_axis(affine)
    ax_names = {0: "i (dim 0)", 1: "j (dim 1)", 2: "k (dim 2)"}
    print(f"  [{case_id}] SI axis: {ax_names[si_axis]}, "
          f"increasing={'superior' if si_up else 'inferior'}")

    masks = {label: (data == label).astype(np.uint8) for label in VERTEBRAE_LABELS}
    present = _present_labels(masks)
    print(f"  [{case_id}] Found {len(present)} vertebrae labels")

    # ── 3. Per-label CC cleanup ──
    print(f"  [{case_id}] Per-label cleanup...")
    masks = per_label_cleanup(masks, args.min_component_voxels, keep_top_k=1)

    # ── 4. Spine outlier removal ──
    print(f"  [{case_id}] Spine-centerline outlier removal...")
    masks = remove_spine_outliers(masks, si_axis, args.max_outlier_deviation)

    # ── 5. Overlap resolution ──
    print(f"  [{case_id}] Resolving overlaps (distance transform)...")
    combined = resolve_overlaps(masks, shape)
    masks = _combined_to_masks(combined)

    # ── 6. Anatomical ordering ──
    print(f"  [{case_id}] Enforcing anatomical ordering...")
    masks = enforce_anatomical_ordering(masks, si_axis, si_up)

    # ── 7. Spine-envelope gap filling ──
    print(f"  [{case_id}] Filling gaps inside spine envelope...")
    masks = fill_spine_gaps(masks, si_axis, shape,
                            gap_closing_radius=args.gap_closing_radius)

    # ── 8. Missing vertebra interpolation ──
    if not args.no_interpolate:
        print(f"  [{case_id}] Interpolating missing vertebrae...")
        masks = interpolate_missing_vertebrae(masks, si_axis, si_up, shape)

    # ── 9. Adaptive morphological regularization ──
    print(f"  [{case_id}] Adaptive morphological regularization...")
    masks = adaptive_morphological_regularization(
        masks, fill_holes=not args.no_fill_holes,
        base_closing=args.closing_size,
    )

    # ── Re-resolve overlaps after morph ops ──
    combined = resolve_overlaps(masks, shape)
    masks = _combined_to_masks(combined)

    # ── 10. Gaussian label smoothing ──
    print(f"  [{case_id}] Gaussian label smoothing (sigma={args.smooth_sigma})...")
    combined = _masks_to_combined(masks, shape)
    combined = smooth_labels_gaussian(combined, sigma=args.smooth_sigma,
                                      min_confidence=0.08)
    masks = _combined_to_masks(combined)

    # ── 12. CT-guided refinement (optional) ──
    if ct_data is not None:
        print(f"  [{case_id}] CT-guided bone-mask refinement...")
        masks = ct_guided_refinement(masks, ct_data, shape,
                                     bone_low=args.bone_low,
                                     bone_high=args.bone_high)

    # ── 13. Sanity checks ──
    print(f"  [{case_id}] Sanity checks...")
    volume_sanity_check(masks, case_id)
    adjacency_check(masks, si_axis, case_id)

    # ── 14. Save ──
    print(f"  [{case_id}] Saving...")
    out_combined = np.zeros(shape, dtype=np.uint8)
    for label in VERTEBRAE_LABELS:
        m = masks.get(label)
        if m is not None:
            out_combined[m > 0] = label

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
        m = masks.get(label)
        if m is None:
            m = np.zeros(shape, dtype=np.uint8)
        nib.save(
            nib.Nifti1Image(m.astype(np.uint8), affine),
            os.path.join(seg_dir, f"{name}.nii.gz"),
        )
    final_present = _present_labels(masks)
    print(f"  [{case_id}] Done -> {out_case_dir}  "
          f"({len(final_present)} labels saved)")


def main():
    p = argparse.ArgumentParser(
        description="Postprocess vertebrae masks from SuPreM inference.",
    )
    # I/O
    p.add_argument("--input_dir", type=str, default="./AbdomenAtlasDemoPredict",
                    help="Dir with case folders containing combined_labels.nii.gz")
    p.add_argument("--output_dir", type=str, default="./AbdomenAtlasDemoPredict_refined",
                    help="Output directory for refined masks")
    p.add_argument("--ct_root_path", type=str, default=None,
                    help="Root of original CT data (for CT-guided refinement). "
                         "Expected structure: <ct_root>/<case_id>/ct.nii.gz")
    # Cleanup
    p.add_argument("--min_component_voxels", type=int, default=100,
                    help="Min voxels to keep a connected component")
    p.add_argument("--max_outlier_deviation", type=float, default=60,
                    help="Max lateral deviation from spine centerline (voxels)")
    # Gap filling
    p.add_argument("--gap_closing_radius", type=int, default=10,
                    help="SI-axis closing radius for spine envelope (voxels)")
    p.add_argument("--no_interpolate", action="store_true",
                    help="Disable missing-vertebra interpolation")
    # Morphology
    p.add_argument("--no_fill_holes", action="store_true",
                    help="Disable binary hole filling")
    p.add_argument("--closing_size", type=int, default=3,
                    help="Base binary closing structure size (0 to disable)")
    # Smoothing
    p.add_argument("--smooth_sigma", type=float, default=1.5,
                    help="Gaussian sigma for label boundary smoothing")
    # CT-guided
    p.add_argument("--bone_low", type=int, default=150,
                    help="Lower HU threshold for bone mask (CT-guided mode)")
    p.add_argument("--bone_high", type=int, default=3000,
                    help="Upper HU threshold for bone mask (CT-guided mode)")
    args = p.parse_args()

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
        print(f"No case directories with combined_labels.nii.gz found in "
              f"{args.input_dir}")
        sys.exit(0)

    print(f"Postprocessing {len(case_dirs)} case(s): "
          f"{args.input_dir} -> {args.output_dir}")
    for case_dir in case_dirs:
        process_case(case_dir, args.output_dir, args,
                     ct_root_path=args.ct_root_path)
    print("All done.")


if __name__ == "__main__":
    main()
