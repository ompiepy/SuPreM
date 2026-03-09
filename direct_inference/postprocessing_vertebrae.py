#!/usr/bin/env python3
"""
Postprocess vertebrae segmentation masks produced by SuPreM inference.

Cleans up common prediction errors: small fragments, overlapping labels,
wrong ordering, gaps between vertebrae, jagged boundaries, etc.
Optionally uses the original CT to trim non-bone voxels.

Usage:
    python postprocessing_vertebrae.py --input_dir ./AbdomenAtlasDemoPredict
    python postprocessing_vertebrae.py --input_dir ./AbdomenAtlasDemoPredict \
        --ct_root_path /path/to/AbdomenAtlasDemo
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

# Label 1 = L5 (most inferior) ... Label 24 = C1 (most superior)
VERT_LABELS = list(range(1, 25))
VERT_NAMES = {
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
# small utilities
# ---------------------------------------------------------------------------

def detect_si_axis(affine):
    """Figure out which voxel axis is superior-inferior from the NIfTI affine.
    Returns (axis_index, True if increasing index = more superior).
    """
    for i, code in enumerate(nib.aff2axcodes(affine)):
        if code == "S":
            return i, True
        if code == "I":
            return i, False
    # fallback: assume dim 2 runs S-I (common for axial CTs)
    return 2, True


def centroid_of(mask):
    if not np.any(mask):
        return None
    ijk = np.array(np.where(mask > 0))
    return tuple(float(ijk[ax].mean()) for ax in range(3))


def nonempty_labels(masks):
    return sorted(l for l in VERT_LABELS if masks.get(l) is not None and np.any(masks[l]))


def merge_masks(masks, shape):
    vol = np.zeros(shape, dtype=np.int32)
    for lab in VERT_LABELS:
        m = masks.get(lab)
        if m is not None and np.any(m):
            vol[m > 0] = lab
    return vol


def split_labels(vol):
    return {lab: (vol == lab).astype(np.uint8) for lab in VERT_LABELS}


# ---------------------------------------------------------------------------
# connected-component cleanup
# ---------------------------------------------------------------------------

def keep_largest_components(binary_mask, k, min_size):
    """Keep the k largest 26-connected components that exceed min_size voxels."""
    cc_labels = cc3d.connected_components(binary_mask.astype(np.uint8), connectivity=26)
    areas = {}
    for comp_id, comp in cc3d.each(cc_labels, binary=True, in_place=True):
        areas[comp_id] = fastremap.foreground(comp)
    ranked = sorted(areas, key=areas.get, reverse=True)

    out = np.zeros_like(binary_mask, dtype=np.uint8)
    for comp_id in ranked[:k]:
        if areas[comp_id] >= min_size:
            out[cc_labels == comp_id] = 1
    return out


def cleanup_per_label(masks, min_voxels):
    """For each vertebra, drop small fragments and keep only the biggest blob."""
    shape = next(iter(masks.values())).shape
    cleaned = {}
    for lab in VERT_LABELS:
        m = masks.get(lab)
        if m is None or not np.any(m):
            cleaned[lab] = np.zeros(shape, dtype=np.uint8)
        else:
            cleaned[lab] = keep_largest_components(m, k=1, min_size=min_voxels)
    return cleaned


# ---------------------------------------------------------------------------
# spine-centerline outlier removal
# ---------------------------------------------------------------------------

def drop_outlier_vertebrae(masks, si_axis, max_dev=60):
    """Remove vertebrae whose centroid is too far off the median spine line
    in the lateral (non-SI) directions."""
    lat_axes = [ax for ax in range(3) if ax != si_axis]
    centroids = {}
    for lab in VERT_LABELS:
        m = masks.get(lab)
        if m is not None and np.any(m):
            c = centroid_of(m)
            if c is not None:
                centroids[lab] = c
    if len(centroids) < 3:
        return masks

    lat_pos = np.array([[centroids[l][ax] for ax in lat_axes] for l in centroids])
    median_lat = np.median(lat_pos, axis=0)

    shape = next(iter(masks.values())).shape
    for lab, c in list(centroids.items()):
        offset = np.array([c[ax] for ax in lat_axes])
        if np.linalg.norm(offset - median_lat) > max_dev:
            print(f"    dropped outlier: {VERT_NAMES.get(lab, lab)}")
            masks[lab] = np.zeros(shape, dtype=np.uint8)
    return masks


# ---------------------------------------------------------------------------
# overlap resolution (distance-transform based)
# ---------------------------------------------------------------------------

def resolve_overlaps(masks, shape):
    """Where two+ labels claim the same voxel, give it to whichever label
    that voxel is deepest inside of (highest EDT value)."""
    count = np.zeros(shape, dtype=np.int32)
    present = []
    for lab in VERT_LABELS:
        m = masks.get(lab)
        if m is not None and np.any(m):
            count += (m > 0).astype(np.int32)
            present.append(lab)

    vol = np.zeros(shape, dtype=np.int32)
    for lab in present:
        vol[masks[lab] > 0] = lab

    contested = count > 1
    if not np.any(contested):
        return vol

    pts = np.array(np.where(contested))
    best_d = np.full(pts.shape[1], -1.0)
    best_l = np.zeros(pts.shape[1], dtype=np.int32)
    for lab in present:
        m = masks[lab]
        hit = m[pts[0], pts[1], pts[2]] > 0
        if not np.any(hit):
            continue
        dt = distance_transform_edt(m > 0)
        d = dt[pts[0], pts[1], pts[2]]
        better = (d > best_d) & hit
        best_d[better] = d[better]
        best_l[better] = lab
    vol[pts[0], pts[1], pts[2]] = best_l
    return vol


# ---------------------------------------------------------------------------
# anatomical ordering
# ---------------------------------------------------------------------------

def fix_label_ordering(masks, si_axis, si_up):
    """Make sure label values increase from inferior to superior.

    Keeps the same *set* of label values that exist in the prediction,
    just reassigns them so spatial position matches anatomical order.
    This way partial scans (e.g. only T1-L5 visible) stay correctly
    labeled instead of being wrongly shifted to start at C1.
    """
    centroids = {}
    for lab in VERT_LABELS:
        m = masks.get(lab)
        if m is not None and np.any(m):
            c = centroid_of(m)
            if c is not None:
                centroids[lab] = c
    if len(centroids) < 2:
        return masks

    sign = 1 if si_up else -1
    by_position = sorted(centroids, key=lambda l: sign * centroids[l][si_axis])
    by_value = sorted(centroids)

    if by_position == by_value:
        return masks  # already fine

    remap = {old: new for old, new in zip(by_position, by_value)}
    shape = next(iter(masks.values())).shape
    out = {lab: np.zeros(shape, dtype=np.uint8) for lab in VERT_LABELS}
    for old, new in remap.items():
        m = masks.get(old)
        if m is not None and np.any(m):
            out[new][m > 0] = 1

    n_swaps = sum(1 for o, n in remap.items() if o != n)
    if n_swaps:
        print(f"    reordered {n_swaps} label(s)")
    return out


# ---------------------------------------------------------------------------
# spine-envelope gap filling
# ---------------------------------------------------------------------------

def fill_gaps_in_spine(masks, si_axis, shape, closing_radius=10):
    """Close gaps between adjacent vertebrae by building a spine envelope
    and assigning unlabeled interior voxels to the nearest label."""
    vol = merge_masks(masks, shape)
    has_label = vol > 0
    if not np.any(has_label):
        return masks

    # elongated closing along the spine to bridge inter-vertebra spaces
    kern = [3, 3, 3]
    kern[si_axis] = closing_radius * 2 + 1
    envelope = ndimage.binary_closing(has_label, structure=np.ones(kern, dtype=bool))
    # small isotropic pass + hole fill to clean up the envelope
    envelope = ndimage.binary_closing(envelope, structure=np.ones((5, 5, 5), dtype=bool))
    envelope = ndimage.binary_fill_holes(envelope)

    gaps = envelope & ~has_label
    n_gap = int(np.sum(gaps))
    if n_gap == 0:
        return masks
    print(f"    filling {n_gap} gap voxels inside spine envelope")

    # assign each gap voxel to the spatially closest vertebra
    gap_pts = np.array(np.where(gaps))
    best_d = np.full(gap_pts.shape[1], np.inf)
    best_l = np.zeros(gap_pts.shape[1], dtype=np.int32)
    for lab in nonempty_labels(masks):
        dt = distance_transform_edt(~(masks[lab] > 0))
        d = dt[gap_pts[0], gap_pts[1], gap_pts[2]]
        closer = d < best_d
        best_d[closer] = d[closer]
        best_l[closer] = lab

    filled = vol.copy()
    filled[gap_pts[0], gap_pts[1], gap_pts[2]] = best_l
    return split_labels(filled)


# ---------------------------------------------------------------------------
# missing vertebra interpolation
# ---------------------------------------------------------------------------

def interpolate_missing(masks, si_axis, si_up, shape):
    """Try to fill in vertebrae that are missing from an otherwise
    contiguous sequence by growing a seed at the interpolated position."""
    present = nonempty_labels(masks)
    if len(present) < 2:
        return masks

    centroids, volumes = {}, {}
    for lab in present:
        c = centroid_of(masks[lab])
        if c is not None:
            centroids[lab] = np.array(c)
            volumes[lab] = int(np.sum(masks[lab] > 0))

    lo, hi = min(present), max(present)
    missing = [l for l in range(lo, hi + 1) if l not in present]
    if not missing:
        return masks

    median_vol = float(np.median(list(volumes.values())))
    struct6 = ndimage.generate_binary_structure(3, 1)

    for lab in missing:
        below = [l for l in present if l < lab]
        above = [l for l in present if l > lab]
        if not below or not above:
            continue
        lo_nb, hi_nb = max(below), min(above)
        if lo_nb not in centroids or hi_nb not in centroids:
            continue

        # linearly interpolate the centroid position
        t = (lab - lo_nb) / (hi_nb - lo_nb)
        est = np.round(centroids[lo_nb] * (1 - t) + centroids[hi_nb] * t).astype(int)
        est = np.clip(est, 0, np.array(shape) - 1)

        target_vol = int(median_vol * 0.7)

        # grow a ball from the estimated centroid
        seed = np.zeros(shape, dtype=np.uint8)
        seed[est[0], est[1], est[2]] = 1
        blob = seed.copy()
        for _ in range(80):
            blob = ndimage.binary_dilation(blob, structure=struct6).astype(np.uint8)
            if int(np.sum(blob)) >= target_vol:
                break

        # trim excess voxels, keeping those closest to centroid
        pts = np.array(np.where(blob > 0))
        if pts.shape[1] > target_vol:
            dists = np.sum((pts.T - est) ** 2, axis=1)
            keep = np.argsort(dists)[:target_vol]
            blob = np.zeros(shape, dtype=np.uint8)
            blob[pts[0, keep], pts[1, keep], pts[2, keep]] = 1

        # don't stomp on existing labels
        blob[merge_masks(masks, shape) > 0] = 0
        if np.any(blob):
            masks[lab] = blob
            print(f"    interpolated {VERT_NAMES.get(lab, lab)}: {int(np.sum(blob))} vox")

    return masks


# ---------------------------------------------------------------------------
# morphological regularization (adaptive kernel per vertebra)
# ---------------------------------------------------------------------------

def morph_regularize(masks, fill_holes=True, base_closing=3):
    """Hole-fill, close, and open each vertebra mask.
    Kernel size scales with vertebra volume so lumbar vertebrae get a
    larger closing kernel than cervical ones.
    """
    shape = next(iter(masks.values())).shape
    present = nonempty_labels(masks)
    if not present:
        return masks

    vols = {l: int(np.sum(masks[l] > 0)) for l in present}
    med_vol = float(np.median(list(vols.values())))

    out = {}
    for lab in VERT_LABELS:
        m = masks.get(lab)
        if m is None or not np.any(m):
            out[lab] = np.zeros(shape, dtype=np.uint8)
            continue

        ratio = vols.get(lab, med_vol) / max(med_vol, 1.0)
        ksz = max(base_closing, int(base_closing * min(ratio, 2.0)))
        ksz = ksz if ksz % 2 == 1 else ksz + 1  # keep odd

        m = m.astype(bool)
        if fill_holes:
            m = ndimage.binary_fill_holes(m)
        m = ndimage.binary_closing(m, structure=np.ones((ksz,) * 3))
        if ksz >= 3:
            m = ndimage.binary_opening(m, structure=np.ones((3, 3, 3)))
        out[lab] = m.astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# gaussian label smoothing
# ---------------------------------------------------------------------------

def smooth_labels(vol, sigma=1.5, min_conf=0.08):
    """Blur each label channel with a Gaussian and pick the winner.
    Gives much smoother inter-vertebra boundaries than raw voxelwise labels.
    min_conf prevents labels from bleeding far into background.
    """
    present = np.unique(vol)
    present = present[present > 0]
    if len(present) == 0:
        return vol

    scores = np.zeros(vol.shape + (len(present),), dtype=np.float32)
    for i, lab in enumerate(present):
        scores[..., i] = ndimage.gaussian_filter(
            (vol == lab).astype(np.float32), sigma=sigma)

    peak = np.max(scores, axis=-1)
    winner = np.argmax(scores, axis=-1)
    result = np.zeros_like(vol)
    mask = peak > min_conf
    result[mask] = present[winner[mask]]
    return result


# ---------------------------------------------------------------------------
# CT-guided bone-mask refinement (optional)
# ---------------------------------------------------------------------------

def refine_with_ct(masks, ct, shape, hu_lo=150, hu_hi=3000, dilate=2):
    """Intersect vertebra masks with a dilated bone mask from the CT.
    Removes soft-tissue false positives while keeping partial-volume
    boundary voxels thanks to the dilation."""
    bone = (ct >= hu_lo) & (ct <= hu_hi)
    if dilate > 0:
        bone = ndimage.binary_dilation(
            bone, structure=ndimage.generate_binary_structure(3, 1),
            iterations=dilate)

    total_trimmed = 0
    out = {}
    for lab in VERT_LABELS:
        m = masks.get(lab)
        if m is None or not np.any(m):
            out[lab] = np.zeros(shape, dtype=np.uint8)
            continue
        refined = (m > 0) & bone
        total_trimmed += int(np.sum(m > 0)) - int(np.sum(refined))
        if np.any(refined):
            out[lab] = keep_largest_components(refined.astype(np.uint8), k=1, min_size=50)
        else:
            out[lab] = m
    if total_trimmed > 0:
        print(f"    CT refinement trimmed {total_trimmed} non-bone voxels")
    return out


# ---------------------------------------------------------------------------
# sanity checks (just prints warnings, doesn't modify anything)
# ---------------------------------------------------------------------------

def check_volumes(masks, case_id):
    vols = [(l, int(np.sum(masks[l] > 0)))
            for l in VERT_LABELS if masks.get(l) is not None and np.any(masks[l])]
    if len(vols) < 2:
        return
    vals = [v for _, v in vols]
    med = float(np.median(vals))
    for lab, v in vols:
        if v > 3 * med or v < 0.3 * med:
            print(f"  [WARN] {case_id} {VERT_NAMES.get(lab, lab)}: "
                  f"vol={v} (median={med:.0f})")


def check_adjacency(masks, si_axis, case_id):
    centroids = {}
    for lab in VERT_LABELS:
        m = masks.get(lab)
        if m is not None and np.any(m):
            c = centroid_of(m)
            if c is not None:
                centroids[lab] = c[si_axis]
    present = sorted(centroids)
    if len(present) < 2:
        return
    gaps = [abs(centroids[b] - centroids[a]) for a, b in zip(present, present[1:])]
    med_gap = float(np.median(gaps))
    if med_gap == 0:
        return
    for i, (a, b) in enumerate(zip(present, present[1:])):
        if gaps[i] > 3 * med_gap:
            print(f"  [WARN] {case_id} large gap between "
                  f"{VERT_NAMES.get(a, a)} and {VERT_NAMES.get(b, b)} "
                  f"({gaps[i]:.0f} vs median {med_gap:.0f})")


# ---------------------------------------------------------------------------
# main per-case pipeline
# ---------------------------------------------------------------------------

def process_case(case_dir, output_dir, args, ct_root=None):
    combined_path = os.path.join(case_dir, "combined_labels.nii.gz")
    if not os.path.isfile(combined_path):
        print(f"  skip {case_dir} (no combined_labels.nii.gz)")
        return
    case_id = os.path.basename(case_dir.rstrip("/"))

    print(f"  [{case_id}] loading...")
    nii = nib.load(combined_path)
    data = np.asarray(nii.dataobj).astype(np.int32)
    shape = data.shape
    affine = nii.affine.copy()

    # optionally load the original CT for bone-mask refinement
    ct = None
    if ct_root:
        ct_path = os.path.join(ct_root, case_id, "ct.nii.gz")
        if os.path.isfile(ct_path):
            print(f"  [{case_id}] loading CT...")
            ct_nii = nib.load(ct_path)
            ct = np.asarray(ct_nii.dataobj).astype(np.float32)
            if ct.shape != shape:
                print(f"  [{case_id}] CT shape {ct.shape} != pred shape {shape}, skipping CT")
                ct = None

    si_axis, si_up = detect_si_axis(affine)
    print(f"  [{case_id}] SI axis = dim {si_axis}, "
          f"{'ascending' if si_up else 'descending'}")

    masks = {lab: (data == lab).astype(np.uint8) for lab in VERT_LABELS}
    print(f"  [{case_id}] {len(nonempty_labels(masks))} labels in raw prediction")

    # -- clean small fragments --
    masks = cleanup_per_label(masks, args.min_component_voxels)

    # -- drop blobs far off the spine line --
    masks = drop_outlier_vertebrae(masks, si_axis, args.max_outlier_deviation)

    # -- resolve overlapping labels --
    vol = resolve_overlaps(masks, shape)
    masks = split_labels(vol)

    # -- fix any ordering violations --
    masks = fix_label_ordering(masks, si_axis, si_up)

    # -- fill spaces between vertebrae inside the spine envelope --
    masks = fill_gaps_in_spine(masks, si_axis, shape,
                               closing_radius=args.gap_closing_radius)

    # -- synthesise any missing vertebrae in the sequence --
    if not args.no_interpolate:
        masks = interpolate_missing(masks, si_axis, si_up, shape)

    # -- per-vertebra morphological regularisation --
    masks = morph_regularize(masks, fill_holes=not args.no_fill_holes,
                             base_closing=args.closing_size)

    # morph ops can re-introduce tiny overlaps, clean them up
    vol = resolve_overlaps(masks, shape)
    masks = split_labels(vol)

    # -- gaussian smoothing of label boundaries --
    vol = merge_masks(masks, shape)
    vol = smooth_labels(vol, sigma=args.smooth_sigma)
    masks = split_labels(vol)

    # -- optional: trim to bone HU range using the CT --
    if ct is not None:
        masks = refine_with_ct(masks, ct, shape,
                               hu_lo=args.bone_low, hu_hi=args.bone_high)

    # -- sanity checks (just warnings) --
    check_volumes(masks, case_id)
    check_adjacency(masks, si_axis, case_id)

    # -- save --
    out_dir = os.path.join(output_dir, case_id)
    os.makedirs(out_dir, exist_ok=True)

    out_vol = np.zeros(shape, dtype=np.uint8)
    for lab in VERT_LABELS:
        m = masks.get(lab)
        if m is not None:
            out_vol[m > 0] = lab
    nib.save(nib.Nifti1Image(out_vol, affine),
             os.path.join(out_dir, "combined_labels.nii.gz"))

    seg_dir = os.path.join(out_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    for lab in VERT_LABELS:
        m = masks.get(lab, np.zeros(shape, dtype=np.uint8))
        nib.save(nib.Nifti1Image(m.astype(np.uint8), affine),
                 os.path.join(seg_dir, f"{VERT_NAMES[lab]}.nii.gz"))

    n_out = len(nonempty_labels(masks))
    print(f"  [{case_id}] saved {n_out} labels -> {out_dir}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Postprocess vertebrae masks from SuPreM inference.")

    p.add_argument("--input_dir", default="./AbdomenAtlasDemoPredict")
    p.add_argument("--output_dir", default="./AbdomenAtlasDemoPredict_refined")
    p.add_argument("--ct_root_path", default=None,
                   help="path to original CTs (<root>/<case>/ct.nii.gz)")

    p.add_argument("--min_component_voxels", type=int, default=100)
    p.add_argument("--max_outlier_deviation", type=float, default=60,
                   help="max lateral offset from spine centerline (voxels)")
    p.add_argument("--gap_closing_radius", type=int, default=10,
                   help="SI-axis closing half-width for spine envelope")
    p.add_argument("--no_interpolate", action="store_true",
                   help="skip missing-vertebra interpolation")
    p.add_argument("--no_fill_holes", action="store_true")
    p.add_argument("--closing_size", type=int, default=3,
                   help="base morphological closing kernel size")
    p.add_argument("--smooth_sigma", type=float, default=1.5,
                   help="gaussian sigma for label smoothing")
    p.add_argument("--bone_low", type=int, default=150,
                   help="lower HU bound for bone mask")
    p.add_argument("--bone_high", type=int, default=3000,
                   help="upper HU bound for bone mask")

    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"error: {args.input_dir} not found")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    cases = sorted(
        os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
        and os.path.isfile(os.path.join(args.input_dir, d, "combined_labels.nii.gz"))
    )
    if not cases:
        print(f"no cases found in {args.input_dir}")
        sys.exit(0)

    print(f"processing {len(cases)} case(s): {args.input_dir} -> {args.output_dir}")
    for case_dir in cases:
        process_case(case_dir, args.output_dir, args, ct_root=args.ct_root_path)
    print("done.")


if __name__ == "__main__":
    main()
