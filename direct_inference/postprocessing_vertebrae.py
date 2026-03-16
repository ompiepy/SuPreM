#!/usr/bin/env python3
"""
Postprocess vertebrae segmentation masks from SuPreM inference.

Applies connected-component cleanup, overlap resolution, optional graph-based
label correction, fragment reassignment, and CT-guided bone masking. Uses
volume priors to split oversized vertebrae and transfer voxels to undersized
neighbors. Pass --ct_root_path when original CTs are available for bone masking.

  python postprocessing_vertebrae.py --input_dir ./pred --output_dir ./refined
  python postprocessing_vertebrae.py --input_dir ./pred --ct_root_path /path/to/ct
"""
import argparse
import heapq
import os
import sys

import numpy as np
import nibabel as nib
import cc3d
import fastremap
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

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

SPINE_LEVEL = {}
for _l in range(1, 6):
    SPINE_LEVEL[_l] = "lumbar"
for _l in range(6, 18):
    SPINE_LEVEL[_l] = "thoracic"
for _l in range(18, 25):
    SPINE_LEVEL[_l] = "cervical"

# Inter-vertebral distance stats (mm) and volume regressors per spine level
IVD_STATS = {
    "cervical": {"mu": 16.77, "sigma": 2.18},
    "thoracic": {"mu": 23.32, "sigma": 3.55},
    "lumbar":   {"mu": 32.68, "sigma": 2.84},
}
VOL_REGRESS = {
    "cervical": {"a_prev": 1.03, "c_prev": 1471, "a_next": 0.92, "c_next":  497},
    "thoracic": {"a_prev": 1.03, "c_prev": 1354, "a_next": 0.94, "c_next": -140},
    "lumbar":   {"a_prev": 1.05, "c_prev":  981, "a_next": 0.94, "c_next": -269},
}

CERVICAL_LABELS = set(range(18, 25))
THORACIC_LABELS = set(range(6, 18))
LUMBAR_LABELS = set(range(1, 6))


# --- Utilities ---

def detect_si_axis(affine):
    """Infer superior-inferior axis from NIfTI affine. Returns (axis_index, si_up)."""
    for i, code in enumerate(nib.aff2axcodes(affine)):
        if code == "S":
            return i, True
        if code == "I":
            return i, False
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


# --- Connected-component cleanup ---

def keep_largest_components(binary_mask, k, min_size):
    """Retain the k largest 26-connected components above min_size voxels."""
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
    """Per label: drop small fragments, keep largest component."""
    shape = next(iter(masks.values())).shape
    cleaned = {}
    for lab in VERT_LABELS:
        m = masks.get(lab)
        if m is None or not np.any(m):
            cleaned[lab] = np.zeros(shape, dtype=np.uint8)
        else:
            cleaned[lab] = keep_largest_components(m, k=1, min_size=min_voxels)
    return cleaned


# --- Fragment reassignment ---

def reassign_adjacent_fragments(masks, si_axis, min_cc_voxels=500):
    """Reassign fragments to the adjacent vertebra whose main body is closest."""
    shape = next(iter(masks.values())).shape
    present = nonempty_labels(masks)
    if len(present) < 2:
        return masks

    centroids = {}
    for lab in present:
        c = centroid_of(masks[lab])
        if c is not None:
            centroids[lab] = np.array(c)

    n_reassigned = 0
    for idx, lab in enumerate(present):
        m = masks[lab]
        if not np.any(m):
            continue

        neighbors = []
        if idx > 0:
            neighbors.append(present[idx - 1])
        if idx < len(present) - 1:
            neighbors.append(present[idx + 1])
        if not neighbors:
            continue

        cc_map = cc3d.connected_components(m.astype(np.uint8), connectivity=26)
        unique_ccs = np.unique(cc_map)
        unique_ccs = unique_ccs[unique_ccs != 0]

        if len(unique_ccs) <= 1:
            continue

        cc_sizes = {cid: int(np.sum(cc_map == cid)) for cid in unique_ccs}
        main_cc = max(cc_sizes, key=cc_sizes.get)

        for cid in unique_ccs:
            if cid == main_cc:
                continue
            if cc_sizes[cid] < min_cc_voxels:
                continue

            frag_coords = np.array(np.where(cc_map == cid))
            frag_center = frag_coords.mean(axis=1)

            if lab not in centroids:
                continue
            dist_self = np.linalg.norm(frag_center - centroids[lab])

            for nb in neighbors:
                if nb not in centroids:
                    continue
                dist_nb = np.linalg.norm(frag_center - centroids[nb])
                if dist_nb < dist_self * 0.8:
                    masks[nb] = masks[nb].copy()
                    masks[nb][cc_map == cid] = 1
                    masks[lab][cc_map == cid] = 0
                    n_reassigned += 1
                    print(f"    reassigned fragment ({cc_sizes[cid]} vox) "
                          f"from {VERT_NAMES.get(lab, lab)} -> {VERT_NAMES.get(nb, nb)}")
                    break

    if n_reassigned:
        print(f"    total fragments reassigned: {n_reassigned}")
    return masks


# --- Balance protrusion ---

def balance_protrusion_pairs(masks, si_axis, si_up, min_cc_voxels=500):
    """Move components that clearly belong to the adjacent vertebra (by SI position)."""
    sign = 1 if si_up else -1
    present = nonempty_labels(masks)
    if len(present) < 2:
        return masks

    centroids_si = {}
    for lab in present:
        c = centroid_of(masks[lab])
        if c is not None:
            centroids_si[lab] = sign * c[si_axis]

    present_sorted = sorted(
        [l for l in present if l in centroids_si],
        key=lambda l: centroids_si[l])

    n_moves = 0
    for i in range(len(present_sorted) - 1):
        lab_a = present_sorted[i]       # inferior
        lab_b = present_sorted[i + 1]   # superior
        si_a = centroids_si[lab_a]
        si_b = centroids_si[lab_b]

        # CCs of A that protrude into B's territory
        cc_a = cc3d.connected_components(masks[lab_a].astype(np.uint8),
                                         connectivity=26)
        for cid in np.unique(cc_a):
            if cid == 0:
                continue
            coords = np.argwhere(cc_a == cid)
            if coords.shape[0] < min_cc_voxels:
                continue
            med_si = sign * float(np.median(coords[:, si_axis]))
            if med_si > si_b:
                masks[lab_b] = np.maximum(
                    masks[lab_b],
                    (cc_a == cid).astype(np.uint8))
                masks[lab_a][cc_a == cid] = 0
                n_moves += 1

        # CCs of B that drop into A's territory
        cc_b = cc3d.connected_components(masks[lab_b].astype(np.uint8),
                                         connectivity=26)
        for cid in np.unique(cc_b):
            if cid == 0:
                continue
            coords = np.argwhere(cc_b == cid)
            if coords.shape[0] < min_cc_voxels:
                continue
            med_si = sign * float(np.median(coords[:, si_axis]))
            if med_si < si_a:
                masks[lab_a] = np.maximum(
                    masks[lab_a],
                    (cc_b == cid).astype(np.uint8))
                masks[lab_b][cc_b == cid] = 0
                n_moves += 1

    if n_moves:
        print(f"    [balance] moved {n_moves} protruding component(s)")
    return masks


# --- Boundary re-optimization (optional) ---

def reoptimize_adjacent_boundaries(masks, si_axis, si_up):
    """Re-split adjacent pairs at the slice with minimum cross-sectional area."""
    present = nonempty_labels(masks)
    if len(present) < 2:
        return masks

    sign = 1 if si_up else -1
    centroids = {}
    for lab in present:
        c = centroid_of(masks[lab])
        if c is not None:
            centroids[lab] = c

    ordered = sorted(
        [l for l in present if l in centroids],
        key=lambda l: sign * centroids[l][si_axis])

    n_resplit = 0

    for i in range(len(ordered) - 1):
        la = ordered[i]        # inferior
        lb = ordered[i + 1]    # superior
        ma, mb = masks[la], masks[lb]

        vol_a = int(np.sum(ma > 0))
        vol_b = int(np.sum(mb > 0))
        if vol_a < 200 or vol_b < 200:
            continue

        ca_si = centroids[la][si_axis]
        cb_si = centroids[lb][si_axis]
        si_lo = min(ca_si, cb_si)
        si_hi = max(ca_si, cb_si)
        gap = si_hi - si_lo
        if gap < 4:
            continue

        margin = gap * 0.2
        zone_lo = int(np.floor(si_lo + margin))
        zone_hi = int(np.ceil(si_hi - margin))
        zone_lo = max(zone_lo, 0)
        zone_hi = min(zone_hi, ma.shape[si_axis] - 1)
        if zone_lo >= zone_hi:
            continue

        combined = (ma > 0) | (mb > 0)

        best_slice = None
        best_area = float('inf')
        for s in range(zone_lo, zone_hi + 1):
            idx = [slice(None)] * 3
            idx[si_axis] = s
            area = int(np.sum(combined[tuple(idx)]))
            if 0 < area < best_area:
                best_area = area
                best_slice = s

        if best_slice is None:
            continue

        all_coords = np.where(combined)
        si_vals = all_coords[si_axis]
        in_zone = (si_vals >= zone_lo) & (si_vals <= zone_hi)
        if not np.any(in_zone):
            continue

        zone_coords = tuple(c[in_zone] for c in all_coords)
        zone_si = si_vals[in_zone]

        if si_up:
            is_inferior = zone_si <= best_slice
        else:
            is_inferior = zone_si >= best_slice

        a_zone = tuple(c[is_inferior] for c in zone_coords)
        b_zone = tuple(c[~is_inferior] for c in zone_coords)

        new_ma = ma.copy()
        new_mb = mb.copy()
        new_ma[zone_coords] = 0
        new_mb[zone_coords] = 0
        new_ma[a_zone] = 1
        new_mb[b_zone] = 1

        new_vol_a = int(np.sum(new_ma > 0))
        new_vol_b = int(np.sum(new_mb > 0))
        total = vol_a + vol_b
        if new_vol_a < 0.15 * total or new_vol_b < 0.15 * total:
            continue

        masks[la] = new_ma
        masks[lb] = new_mb

        c_new_a = centroid_of(new_ma)
        c_new_b = centroid_of(new_mb)
        if c_new_a is not None:
            centroids[la] = c_new_a
        if c_new_b is not None:
            centroids[lb] = c_new_b

        n_resplit += 1

    if n_resplit:
        print(f"    [boundary] re-optimized {n_resplit} adjacent pair(s)")
    return masks


# --- Overlap resolution ---

def resolve_overlaps(masks, shape):
    """Resolve overlaps by assigning each contested voxel to the label it is deepest inside (EDT)."""
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


# --- Graph-based label re-identification ---

def _slot_group(label):
    """Return spine level (lumbar/thoracic/cervical) for label id."""
    if label in LUMBAR_LABELS:
        return "lumbar"
    if label in THORACIC_LABELS:
        return "thoracic"
    return "cervical"


def graph_relabel(masks, si_axis, si_up, affine):
    """Reassign labels via shortest path over (slot, label) with position/volume/distance costs."""
    present = nonempty_labels(masks)
    n = len(present)
    if n < 3:
        return masks

    spacing = get_voxel_spacing(affine)
    si_spacing = float(spacing[si_axis])
    sign = 1.0 if si_up else -1.0

    blob_info = []
    for lab in present:
        c = centroid_of(masks[lab])
        if c is None:
            continue
        blob_info.append({
            "orig_label": lab,
            "centroid": c,
            "si_pos": sign * c[si_axis],
            "volume": int(np.sum(masks[lab] > 0)),
        })
    blob_info.sort(key=lambda b: b["si_pos"])
    n = len(blob_info)
    if n < 3:
        return masks

    volumes = [b["volume"] for b in blob_info]
    si_positions_mm = [b["si_pos"] * si_spacing for b in blob_info]

    inter_dists_mm = []
    for i in range(n - 1):
        inter_dists_mm.append(abs(si_positions_mm[i + 1] - si_positions_mm[i]))

    N_LABELS = 24
    INF = 1e9
    unary = np.full((n, N_LABELS), INF, dtype=np.float64)

    for slot_idx in range(n):
        for cand_label in VERT_LABELS:
            cand_idx = cand_label - 1
            cost = 0.0

            frac_slot = slot_idx / max(n - 1, 1)
            frac_label = cand_idx / (N_LABELS - 1)
            cost += 3.0 * (frac_slot - frac_label) ** 2

            level = _slot_group(cand_label)
            reg = VOL_REGRESS[level]
            vol_predictions = []
            if slot_idx > 0:
                vol_predictions.append(
                    reg["a_prev"] * volumes[slot_idx - 1] + reg["c_prev"])
            if slot_idx < n - 1:
                vol_predictions.append(
                    reg["a_next"] * volumes[slot_idx + 1] + reg["c_next"])
            if vol_predictions:
                expected_vol = float(np.mean(vol_predictions))
                if expected_vol > 0:
                    ratio = volumes[slot_idx] / expected_vol
                    cost += 0.5 * (np.log(max(ratio, 0.1))) ** 2

            stats = IVD_STATS[level]
            if slot_idx < n - 1:
                d = inter_dists_mm[slot_idx]
                z_score = (d - stats["mu"]) / max(stats["sigma"], 1.0)
                cost += 0.3 * min(z_score ** 2, 25.0)

            if frac_slot < 0.25 and cand_label >= 18:
                cost += 5.0
            elif frac_slot > 0.75 and cand_label <= 5:
                cost += 5.0

            unary[slot_idx, cand_idx] = cost

    SRC = (-1, -1)
    DST = (-2, -2)
    dist = {SRC: 0.0}
    prev = {SRC: None}
    heap = [(0.0, SRC)]

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > dist.get(u, INF):
            continue

        if u == DST:
            break

        if u == SRC:
            for li in range(N_LABELS):
                v = (0, li)
                d_v = d_u + unary[0, li]
                if d_v < dist.get(v, INF):
                    dist[v] = d_v
                    prev[v] = u
                    heapq.heappush(heap, (d_v, v))
        else:
            slot, li = u
            if slot == n - 1:
                v = DST
                d_v = d_u
                if d_v < dist.get(v, INF):
                    dist[v] = d_v
                    prev[v] = u
                    heapq.heappush(heap, (d_v, v))
            else:
                next_slot = slot + 1
                next_li = li + 1
                if next_li < N_LABELS:
                    v = (next_slot, next_li)
                    d_v = d_u + unary[next_slot, next_li]
                    if d_v < dist.get(v, INF):
                        dist[v] = d_v
                        prev[v] = u
                        heapq.heappush(heap, (d_v, v))
                next_li2 = li + 2
                if next_li2 < N_LABELS:
                    v = (next_slot, next_li2)
                    skip_penalty = 2.0
                    d_v = d_u + unary[next_slot, next_li2] + skip_penalty
                    if d_v < dist.get(v, INF):
                        dist[v] = d_v
                        prev[v] = u
                        heapq.heappush(heap, (d_v, v))

    path = []
    node = DST
    while node is not None and node != SRC:
        if node != DST:
            path.append(node)
        node = prev.get(node)
    path.reverse()

    if len(path) != n:
        print(f"    [graph] path length {len(path)} != {n} slots, skipping relabel")
        return masks

    new_labels = [VERT_LABELS[li] for _, li in path]
    old_labels = [b["orig_label"] for b in blob_info]

    changes = sum(1 for o, nl in zip(old_labels, new_labels) if o != nl)
    if changes == 0:
        print("    [graph] labels already optimal, no changes")
        return masks

    print(f"    [graph] relabelling {changes}/{n} slots:")
    shape = next(iter(masks.values())).shape
    new_masks = {lab: np.zeros(shape, dtype=np.uint8) for lab in VERT_LABELS}

    for slot_idx, (old_lab, new_lab) in enumerate(zip(old_labels, new_labels)):
        m = masks[old_lab]
        if old_lab != new_lab:
            print(f"      slot {slot_idx}: {VERT_NAMES.get(old_lab, old_lab)} -> "
                  f"{VERT_NAMES.get(new_lab, new_lab)}")
        new_masks[new_lab] = np.maximum(new_masks[new_lab], m)

    return new_masks


# --- Label reordering ---

def fix_label_ordering(masks, si_axis, si_up):
    """Swap adjacent pairs when their SI centroids are reversed beyond half the median gap."""
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
    present = sorted(centroids.keys())

    si_positions = {lab: sign * centroids[lab][si_axis] for lab in present}
    gaps = []
    for i in range(len(present) - 1):
        gaps.append(abs(si_positions[present[i + 1]] - si_positions[present[i]]))
    if not gaps:
        return masks
    median_gap = float(np.median(gaps))
    if median_gap < 1.0:
        return masks

    swap_threshold = median_gap * 0.5

    shape = next(iter(masks.values())).shape
    n_swaps = 0
    for i in range(len(present) - 1):
        a, b = present[i], present[i + 1]
        if si_positions[a] > si_positions[b] + swap_threshold:
            masks[a], masks[b] = masks[b], masks[a]
            centroids[a], centroids[b] = centroids[b], centroids[a]
            si_positions[a], si_positions[b] = si_positions[b], si_positions[a]
            n_swaps += 1

    if n_swaps:
        print(f"    swapped {n_swaps} adjacent label pair(s)")
    return masks


# --- Statistical validation and splitting ---

def get_voxel_spacing(affine):
    """Voxel spacing (mm) from affine."""
    return np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))


def validate_and_split_fused(masks, si_axis, si_up, affine):
    """Flag volume/distance violations; split oversized vertebrae and transfer to undersized neighbors."""
    present = nonempty_labels(masks)
    if len(present) < 3:
        return masks

    spacing = get_voxel_spacing(affine)
    si_spacing = spacing[si_axis]

    centroids, volumes = {}, {}
    for lab in present:
        c = centroid_of(masks[lab])
        if c is not None:
            centroids[lab] = c
            volumes[lab] = int(np.sum(masks[lab] > 0))

    sign = 1 if si_up else -1

    for i, lab in enumerate(present):
        if lab not in volumes:
            continue
        level = SPINE_LEVEL.get(lab, "thoracic")
        vol = volumes[lab]

        predicted_vols = []
        if i > 0 and present[i - 1] in volumes:
            prev_lab = present[i - 1]
            prev_level = SPINE_LEVEL.get(prev_lab, level)
            reg = VOL_REGRESS[prev_level]
            predicted_vols.append(reg["a_prev"] * volumes[prev_lab] + reg["c_prev"])
        if i < len(present) - 1 and present[i + 1] in volumes:
            next_lab = present[i + 1]
            next_level = SPINE_LEVEL.get(next_lab, level)
            reg = VOL_REGRESS[next_level]
            predicted_vols.append(reg["a_next"] * volumes[next_lab] + reg["c_next"])

        if predicted_vols:
            expected = float(np.mean(predicted_vols))
            if expected > 0 and vol > 2.0 * expected:
                print(f"    [WARN] {VERT_NAMES.get(lab, lab)} oversized: "
                      f"{vol} vox vs expected ~{expected:.0f}")
                masks = _try_split_vertebra(masks, lab, si_axis, si_up)
            elif expected > 0 and vol < 0.3 * expected:
                print(f"    [WARN] {VERT_NAMES.get(lab, lab)} undersized: "
                      f"{vol} vox vs expected ~{expected:.0f}")

    already_split = set()
    for i, lab in enumerate(present):
        if lab not in volumes:
            continue
        cur_vol = int(np.sum(masks[lab] > 0))
        level = SPINE_LEVEL.get(lab, "thoracic")

        predicted_vols2 = []
        if i > 0 and present[i - 1] in volumes:
            reg2 = VOL_REGRESS[SPINE_LEVEL.get(present[i - 1], level)]
            predicted_vols2.append(
                reg2["a_prev"] * int(np.sum(masks[present[i - 1]] > 0)) + reg2["c_prev"])
        if i < len(present) - 1 and present[i + 1] in volumes:
            reg2 = VOL_REGRESS[SPINE_LEVEL.get(present[i + 1], level)]
            predicted_vols2.append(
                reg2["a_next"] * int(np.sum(masks[present[i + 1]] > 0)) + reg2["c_next"])

        if predicted_vols2:
            expected2 = float(np.mean(predicted_vols2))
            if expected2 > 0 and cur_vol < 0.3 * expected2:
                for ni in [i - 1, i + 1]:
                    if ni < 0 or ni >= len(present):
                        continue
                    nb = present[ni]
                    if nb in already_split:
                        continue
                    nb_cur = int(np.sum(masks[nb] > 0))
                    nb_level = SPINE_LEVEL.get(nb, "thoracic")
                    nb_preds = []
                    if ni > 0 and present[ni - 1] in volumes:
                        rr = VOL_REGRESS[SPINE_LEVEL.get(present[ni - 1], nb_level)]
                        nb_preds.append(
                            rr["a_prev"] * int(np.sum(masks[present[ni - 1]] > 0)) + rr["c_prev"])
                    if ni < len(present) - 1 and present[ni + 1] in volumes:
                        rr = VOL_REGRESS[SPINE_LEVEL.get(present[ni + 1], nb_level)]
                        nb_preds.append(
                            rr["a_next"] * int(np.sum(masks[present[ni + 1]] > 0)) + rr["c_next"])
                    if not nb_preds:
                        continue
                    nb_expected = float(np.mean(nb_preds))
                    if nb_expected > 0 and nb_cur > 1.5 * nb_expected:
                        print(f"    [WARN] {VERT_NAMES.get(lab, lab)} undersized + "
                              f"{VERT_NAMES.get(nb, nb)} oversized -> splitting neighbor")
                        masks = _try_split_vertebra(masks, nb, si_axis, si_up)
                        already_split.add(nb)
                        break

    for i in range(len(present) - 1):
        a, b = present[i], present[i + 1]
        if a not in centroids or b not in centroids:
            continue
        dist_vox = abs(centroids[b][si_axis] - centroids[a][si_axis])
        dist_mm = dist_vox * si_spacing
        level = SPINE_LEVEL.get(a, "thoracic")
        stats = IVD_STATS[level]
        if dist_mm > stats["mu"] + 3 * stats["sigma"]:
            print(f"    [WARN] large gap {VERT_NAMES.get(a, a)}->{VERT_NAMES.get(b, b)}: "
                  f"{dist_mm:.1f}mm (expected {stats['mu']:.1f}+/-{stats['sigma']:.1f}mm)")

    return masks


def _try_split_vertebra(masks, lab, si_axis, si_up):
    """Split oversized vertebra: transfer voxels to the neediest neighbor by distance to its centroid."""
    m = masks[lab]
    if not np.any(m):
        return masks

    present = nonempty_labels(masks)
    idx = present.index(lab) if lab in present else -1
    if idx < 0:
        return masks

    vol_lab = int(np.sum(m > 0))
    sign = 1 if si_up else -1

    best_nb = None
    best_deficit = 0

    for ni in [idx - 1, idx + 1]:
        if ni < 0 or ni >= len(present):
            continue
        nb = present[ni]
        nb_vol = int(np.sum(masks[nb] > 0))
        nb_level = SPINE_LEVEL.get(nb, "thoracic")
        reg = VOL_REGRESS[nb_level]

        nb_expected_parts = []
        if ni > 0 and present[ni - 1] in present:
            pv = int(np.sum(masks[present[ni - 1]] > 0))
            nb_expected_parts.append(reg["a_prev"] * pv + reg["c_prev"])
        if ni < len(present) - 1 and present[ni + 1] in present:
            nv = int(np.sum(masks[present[ni + 1]] > 0))
            nb_expected_parts.append(reg["a_next"] * nv + reg["c_next"])

        if not nb_expected_parts:
            continue
        nb_expected = float(np.mean(nb_expected_parts))
        if nb_expected <= 0:
            continue

        deficit = nb_expected - nb_vol
        if deficit > best_deficit:
            best_deficit = deficit
            best_nb = nb

    if best_nb is None:
        if idx > 0:
            best_nb = present[idx - 1]
        elif idx < len(present) - 1:
            best_nb = present[idx + 1]
        else:
            return masks

    lab_level = SPINE_LEVEL.get(lab, "thoracic")
    lab_reg = VOL_REGRESS[lab_level]
    lab_expected_parts = []
    if idx > 0:
        pv = int(np.sum(masks[present[idx - 1]] > 0))
        lab_expected_parts.append(lab_reg["a_prev"] * pv + lab_reg["c_prev"])
    if idx < len(present) - 1:
        nv = int(np.sum(masks[present[idx + 1]] > 0))
        lab_expected_parts.append(lab_reg["a_next"] * nv + lab_reg["c_next"])
    lab_expected = float(np.mean(lab_expected_parts)) if lab_expected_parts else vol_lab * 0.5

    surplus = max(0, vol_lab - lab_expected)
    n_transfer = int(min(surplus, best_deficit, vol_lab * 0.55))
    n_transfer = max(n_transfer, int(vol_lab * 0.3))

    if n_transfer < 100:
        return masks

    coords = np.array(np.where(m > 0))
    nb_centroid = centroid_of(masks[best_nb])
    if nb_centroid is None:
        nb_centroid = centroid_of(m)
        if nb_centroid is None:
            return masks

    dists = np.sum((coords.T - np.array(nb_centroid)) ** 2, axis=1)
    order = np.argsort(dists)
    to_transfer = order[:n_transfer]
    to_keep = order[n_transfer:]

    if len(to_keep) < 200:
        return masks

    shape = m.shape
    new_lab = np.zeros(shape, dtype=np.uint8)
    new_lab[coords[0, to_keep], coords[1, to_keep], coords[2, to_keep]] = 1

    donated = np.zeros(shape, dtype=np.uint8)
    donated[coords[0, to_transfer], coords[1, to_transfer], coords[2, to_transfer]] = 1

    masks[lab] = new_lab
    masks[best_nb] = np.maximum(masks[best_nb], donated)

    print(f"    split {VERT_NAMES.get(lab, lab)}: kept {int(np.sum(new_lab))} vox, "
          f"gave {n_transfer} vox to {VERT_NAMES.get(best_nb, best_nb)}")
    return masks


# --- Hole filling ---

def fill_holes_per_label(masks):
    """Fill holes inside each label mask (binary_fill_holes only)."""
    shape = next(iter(masks.values())).shape
    out = {}
    for lab in VERT_LABELS:
        m = masks.get(lab)
        if m is None or not np.any(m):
            out[lab] = np.zeros(shape, dtype=np.uint8)
            continue
        filled = ndimage.binary_fill_holes(m.astype(bool))
        out[lab] = filled.astype(np.uint8)
    return out


# --- Label smoothing ---

def smooth_labels(vol, sigma=0.5, min_conf=0.2):
    """Gaussian blur per label, then assign each voxel to the highest score above min_conf."""
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


# --- CT-guided refinement ---

def refine_boundaries_with_ct(masks, ct, si_axis, si_up, disc_hu_lo=50, disc_hu_hi=200):
    """Reassign boundary voxels between adjacent pairs using EDT from bone-intensity cores."""
    present = nonempty_labels(masks)
    if len(present) < 2 or ct is None:
        return masks

    sign = 1 if si_up else -1
    centroids_si = {}
    for lab in present:
        c = centroid_of(masks[lab])
        if c is not None:
            centroids_si[lab] = sign * c[si_axis]
    ordered = sorted(
        [l for l in present if l in centroids_si],
        key=lambda l: centroids_si[l])

    n_refined = 0
    for i in range(len(ordered) - 1):
        la, lb = ordered[i], ordered[i + 1]
        ma, mb = masks[la], masks[lb]
        combined = (ma > 0) | (mb > 0)
        if not np.any(combined):
            continue

        ca = centroid_of(ma)
        cb = centroid_of(mb)
        if ca is None or cb is None:
            continue

        lo_si = min(ca[si_axis], cb[si_axis])
        hi_si = max(ca[si_axis], cb[si_axis])
        mid_margin = (hi_si - lo_si) * 0.3
        slab_lo = lo_si + mid_margin
        slab_hi = hi_si - mid_margin
        if slab_lo >= slab_hi:
            continue

        si_grid = np.arange(combined.shape[si_axis])
        expand = [1, 1, 1]
        expand[si_axis] = combined.shape[si_axis]
        si_coords = si_grid.reshape(expand)
        si_coords = np.broadcast_to(si_coords, combined.shape)

        slab_mask = (si_coords >= slab_lo) & (si_coords <= slab_hi)
        boundary_zone = combined & slab_mask

        if int(np.sum(boundary_zone)) < 10:
            continue

        bone_a = (ma > 0) & (ct >= 200) & ~slab_mask
        bone_b = (mb > 0) & (ct >= 200) & ~slab_mask
        if not np.any(bone_a) or not np.any(bone_b):
            continue

        dt_a = distance_transform_edt(~bone_a)
        dt_b = distance_transform_edt(~bone_b)

        reassign = boundary_zone & ((dt_a != dt_b) | (ma > 0) | (mb > 0))
        pts = np.where(reassign)
        closer_a = dt_a[pts] <= dt_b[pts]

        old_a = int(np.sum(ma[pts]))
        masks[la][pts] = np.where(closer_a, 1, 0).astype(np.uint8)
        masks[lb][pts] = np.where(~closer_a, 1, 0).astype(np.uint8)
        new_a = int(np.sum(masks[la][pts]))
        moved = abs(new_a - old_a)
        if moved > 0:
            n_refined += 1

    if n_refined:
        print(f"    [IVD boundary] refined {n_refined} adjacent pair(s)")
    return masks


def refine_with_ct(masks, ct, shape, hu_lo=150, hu_hi=3000, dilate=2, max_trim_frac=0.30):
    """Intersect masks with dilated CT bone mask; skip if trim would exceed max_trim_frac."""
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
        orig_count = int(np.sum(m > 0))
        refined = (m > 0) & bone
        refined_count = int(np.sum(refined))
        if orig_count > 0 and (orig_count - refined_count) > max_trim_frac * orig_count:
            out[lab] = m
            continue
        total_trimmed += orig_count - refined_count
        if np.any(refined):
            out[lab] = keep_largest_components(refined.astype(np.uint8), k=1, min_size=50)
        else:
            out[lab] = m
    if total_trimmed > 0:
        print(f"    CT refinement trimmed {total_trimmed} non-bone voxels")
    return out


# --- Optional: outlier removal ---

def drop_outlier_vertebrae(masks, si_axis, max_dev=120):
    """Remove vertebrae whose lateral position deviates from the local spine curve."""
    lat_axes = [ax for ax in range(3) if ax != si_axis]
    centroids = {}
    for lab in VERT_LABELS:
        m = masks.get(lab)
        if m is not None and np.any(m):
            c = centroid_of(m)
            if c is not None:
                centroids[lab] = c
    if len(centroids) < 5:
        return masks

    present = sorted(centroids.keys())
    shape = next(iter(masks.values())).shape
    dropped = []

    for idx, lab in enumerate(present):
        window = []
        for j in range(max(0, idx - 2), min(len(present), idx + 3)):
            if j != idx:
                window.append(present[j])
        if len(window) < 2:
            continue

        local_lat = np.array([[centroids[w][ax] for ax in lat_axes] for w in window])
        local_center = np.mean(local_lat, axis=0)

        my_lat = np.array([centroids[lab][ax] for ax in lat_axes])
        deviation = np.linalg.norm(my_lat - local_center)

        if deviation > max_dev:
            dropped.append(lab)
            masks[lab] = np.zeros(shape, dtype=np.uint8)
            print(f"    dropped outlier: {VERT_NAMES.get(lab, lab)} "
                  f"(dev={deviation:.1f} > {max_dev})")

    return masks


# --- Optional: gap fill and interpolation ---

def fill_gaps_in_spine(masks, si_axis, shape, closing_radius=3):
    """Close spine envelope and assign gap voxels to nearest label."""
    vol = merge_masks(masks, shape)
    has_label = vol > 0
    if not np.any(has_label):
        return masks

    kern = [3, 3, 3]
    kern[si_axis] = closing_radius * 2 + 1
    envelope = ndimage.binary_closing(has_label, structure=np.ones(kern, dtype=bool))
    envelope = ndimage.binary_closing(envelope, structure=np.ones((3, 3, 3), dtype=bool))
    envelope = ndimage.binary_fill_holes(envelope)

    gaps = envelope & ~has_label
    n_gap = int(np.sum(gaps))
    if n_gap == 0:
        return masks
    print(f"    filling {n_gap} gap voxels inside spine envelope")

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


def interpolate_missing(masks, si_axis, si_up, shape):
    """Fill missing labels by growing a blob from the interpolated neighbor position."""
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

        t = (lab - lo_nb) / (hi_nb - lo_nb)
        est = np.round(centroids[lo_nb] * (1 - t) + centroids[hi_nb] * t).astype(int)
        est = np.clip(est, 0, np.array(shape) - 1)

        target_vol = int(median_vol * 0.7)

        seed = np.zeros(shape, dtype=np.uint8)
        seed[est[0], est[1], est[2]] = 1
        blob = seed.copy()
        for _ in range(80):
            blob = ndimage.binary_dilation(blob, structure=struct6).astype(np.uint8)
            if int(np.sum(blob)) >= target_vol:
                break

        pts = np.array(np.where(blob > 0))
        if pts.shape[1] > target_vol:
            dists = np.sum((pts.T - est) ** 2, axis=1)
            keep = np.argsort(dists)[:target_vol]
            blob = np.zeros(shape, dtype=np.uint8)
            blob[pts[0, keep], pts[1, keep], pts[2, keep]] = 1

        blob[merge_masks(masks, shape) > 0] = 0
        if np.any(blob):
            masks[lab] = blob
            print(f"    interpolated {VERT_NAMES.get(lab, lab)}: {int(np.sum(blob))} vox")

    return masks


# --- Warnings ---

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


# --- Per-case pipeline ---

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
    initial_labels = len(nonempty_labels(masks))
    print(f"  [{case_id}] {initial_labels} labels in raw prediction")

    masks = cleanup_per_label(masks, args.min_component_voxels)
    print(f"  [{case_id}] after CC cleanup: {len(nonempty_labels(masks))} labels")

    vol = resolve_overlaps(masks, shape)
    masks = split_labels(vol)

    if not args.no_graph_relabel:
        masks = graph_relabel(masks, si_axis, si_up, affine)
        vol = resolve_overlaps(masks, shape)
        masks = split_labels(vol)

    if args.enable_boundary_reopt:
        for _ in range(args.boundary_iters):
            masks = reoptimize_adjacent_boundaries(masks, si_axis, si_up)
        vol = resolve_overlaps(masks, shape)
        masks = split_labels(vol)

    if not args.no_balance:
        masks = balance_protrusion_pairs(masks, si_axis, si_up,
                                         min_cc_voxels=args.reassign_min_cc)
        vol = resolve_overlaps(masks, shape)
        masks = split_labels(vol)

    if not args.no_reassign:
        masks = reassign_adjacent_fragments(masks, si_axis, min_cc_voxels=args.reassign_min_cc)
        vol = resolve_overlaps(masks, shape)
        masks = split_labels(vol)

    if not args.no_reorder:
        masks = fix_label_ordering(masks, si_axis, si_up)

    if ct is not None and not args.no_ivd_refine:
        masks = refine_boundaries_with_ct(masks, ct, si_axis, si_up)
        vol = resolve_overlaps(masks, shape)
        masks = split_labels(vol)

    if ct is not None:
        masks = refine_with_ct(masks, ct, shape,
                               hu_lo=args.bone_low, hu_hi=args.bone_high)

    if not args.no_validate:
        masks = validate_and_split_fused(masks, si_axis, si_up, affine)
        vol = resolve_overlaps(masks, shape)
        masks = split_labels(vol)

    if not args.no_fill_holes:
        masks = fill_holes_per_label(masks)
        vol = resolve_overlaps(masks, shape)
        masks = split_labels(vol)

    if args.smooth_sigma > 0:
        vol = merge_masks(masks, shape)
        vol = smooth_labels(vol, sigma=args.smooth_sigma, min_conf=args.smooth_min_conf)
        masks = split_labels(vol)

    if args.enable_outlier_removal:
        masks = drop_outlier_vertebrae(masks, si_axis, args.max_outlier_deviation)

    if args.enable_gap_fill:
        masks = fill_gaps_in_spine(masks, si_axis, shape,
                                   closing_radius=args.gap_closing_radius)

    if args.enable_interpolation:
        masks = interpolate_missing(masks, si_axis, si_up, shape)

    check_volumes(masks, case_id)
    check_adjacency(masks, si_axis, case_id)

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


def main():
    p = argparse.ArgumentParser(description="Postprocess vertebrae segmentation from SuPreM.")
    p.add_argument("--input_dir", default="./AbdomenAtlasDemoPredict", help="Input prediction dir")
    p.add_argument("--output_dir", default="./AbdomenAtlasDemoPredict_refined", help="Output dir")
    p.add_argument("--ct_root_path", default=None, help="CT root (<root>/<case>/ct.nii.gz)")

    p.add_argument("--min_component_voxels", type=int, default=100)
    p.add_argument("--reassign_min_cc", type=int, default=500, help="Min voxels for fragment reassign")

    p.add_argument("--no_graph_relabel", action="store_true", help="Skip graph label re-ID")
    p.add_argument("--enable_boundary_reopt", action="store_true", help="Use min-cross-section boundary reopt")
    p.add_argument("--boundary_iters", type=int, default=2)
    p.add_argument("--no_balance", action="store_true", help="Skip protrusion balance")
    p.add_argument("--no_reassign", action="store_true", help="Skip fragment reassignment")
    p.add_argument("--no_reorder", action="store_true", help="Skip label reordering")
    p.add_argument("--no_validate", action="store_true", help="Skip validation/splitting")
    p.add_argument("--no_fill_holes", action="store_true", help="Skip hole filling")
    p.add_argument("--no_ivd_refine", action="store_true", help="Skip CT IVD boundary refine")

    p.add_argument("--smooth_sigma", type=float, default=0.5, help="Smoothing sigma (0=off)")
    p.add_argument("--smooth_min_conf", type=float, default=0.2)

    p.add_argument("--bone_low", type=int, default=150, help="Bone HU lower bound")
    p.add_argument("--bone_high", type=int, default=3000, help="Bone HU upper bound")

    p.add_argument("--enable_outlier_removal", action="store_true")
    p.add_argument("--max_outlier_deviation", type=float, default=120)
    p.add_argument("--enable_gap_fill", action="store_true")
    p.add_argument("--gap_closing_radius", type=int, default=3)
    p.add_argument("--enable_interpolation", action="store_true")

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
