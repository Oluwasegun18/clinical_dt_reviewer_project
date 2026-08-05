
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence

import torch

StateDict = Dict[str, torch.Tensor]
Submissions = Dict[int, StateDict]


def _check_submissions(submissions: Submissions) -> None:
    if not submissions:
        raise ValueError("No submissions were provided for aggregation.")


def _float_keys(state: StateDict) -> List[str]:
    return [k for k, v in state.items() if torch.is_tensor(v) and v.is_floating_point()]


def _copy_template(submissions: Submissions) -> StateDict:
    _check_submissions(submissions)
    return copy.deepcopy(next(iter(submissions.values())))


def _vectorize_state(
    state: StateDict,
    *,
    keys: Optional[Sequence[str]] = None,
    reference_state: Optional[StateDict] = None,
) -> torch.Tensor:
    if keys is None:
        keys = _float_keys(state)
    parts = []
    for k in keys:
        v = state[k].detach().float().cpu()
        if reference_state is not None:
            v = v - reference_state[k].detach().float().cpu()
        parts.append(v.reshape(-1))
    return torch.cat(parts).double() if parts else torch.empty(0, dtype=torch.float64)


def _normalize_weights(cids: List[int], weights: Optional[Dict[int, float]]) -> torch.Tensor:
    if weights is None:
        return torch.ones(len(cids), dtype=torch.float64) / float(len(cids))
    vals = torch.tensor([max(0.0, float(weights.get(cid, 0.0))) for cid in cids], dtype=torch.float64)
    s = float(vals.sum())
    if s <= 1e-12:
        vals[:] = 1.0 / float(len(cids))
    else:
        vals /= s
    return vals


def fedavg_aggregate(
    submissions: Submissions,
    *,
    weights: Optional[Dict[int, float]] = None,
    keys: Optional[Sequence[str]] = None,
) -> StateDict:
    """Weighted FedAvg over floating tensors. Non-floating buffers are copied from the first submission."""
    _check_submissions(submissions)
    cids = sorted(submissions.keys())
    template = _copy_template(submissions)
    if keys is None:
        keys = _float_keys(template)

    w = _normalize_weights(cids, weights)
    for k in keys:
        acc = torch.zeros_like(template[k], dtype=torch.float32)
        for i, cid in enumerate(cids):
            acc += submissions[cid][k].detach().float().cpu() * float(w[i])
        template[k] = acc.to(dtype=template[k].dtype)
    return template


def coordinate_median_aggregate(submissions: Submissions, *, keys: Optional[Sequence[str]] = None) -> StateDict:
    _check_submissions(submissions)
    cids = sorted(submissions.keys())
    template = _copy_template(submissions)
    if keys is None:
        keys = _float_keys(template)

    for k in keys:
        stack = torch.stack([submissions[cid][k].detach().float().cpu() for cid in cids], dim=0)
        template[k] = torch.median(stack, dim=0).values.to(dtype=template[k].dtype)
    return template


def trimmed_mean_aggregate(
    submissions: Submissions,
    *,
    f: Optional[int] = None,
    trim_ratio: Optional[float] = None,
    keys: Optional[Sequence[str]] = None,
) -> StateDict:
    """Coordinate-wise trimmed mean. Prefer f as the estimated number of Byzantine clients."""
    _check_submissions(submissions)
    cids = sorted(submissions.keys())
    n = len(cids)
    template = _copy_template(submissions)
    if keys is None:
        keys = _float_keys(template)

    if f is None:
        ratio = 0.0 if trim_ratio is None else max(0.0, min(0.49, float(trim_ratio)))
        f = int(math.floor(ratio * n))
    f = int(max(0, min(int(f), (n - 1) // 2)))
    if n - 2 * f <= 0:
        f = 0

    for k in keys:
        stack = torch.stack([submissions[cid][k].detach().float().cpu() for cid in cids], dim=0)
        if f > 0:
            stack = torch.sort(stack, dim=0).values[f:n - f]
        template[k] = stack.mean(dim=0).to(dtype=template[k].dtype)
    return template


def _safe_f_for_krum(n: int, f: int) -> int:
    """Krum requires n > 2f + 2, so f <= floor((n-3)/2)."""
    if n < 3:
        return 0
    return max(0, min(int(f), (n - 3) // 2))


def krum_scores(
    submissions: Submissions,
    *,
    f: int = 1,
    reference_state: Optional[StateDict] = None,
    keys: Optional[Sequence[str]] = None,
    distance: str = "squared_l2",
) -> Dict[int, float]:
    """
    Krum score over the n-f-2 closest other submissions.

    distance:
        "squared_l2" : canonical Blanchard et al. Krum score, sum ||x_i-x_j||_2^2.
        "l2"         : Euclidean-distance variant, sum ||x_i-x_j||_2.

    Note:
        The original Krum paper uses squared L2 norm in the score. The "l2"
        option is provided as an experimental variant only.
    """
    _check_submissions(submissions)
    cids = sorted(submissions.keys())
    n = len(cids)
    if keys is None:
        keys = _float_keys(submissions[cids[0]])

    f = _safe_f_for_krum(n, f)
    nb = n - f - 2
    if nb <= 0:
        return {cid: 0.0 for cid in cids}

    vecs = [_vectorize_state(submissions[cid], keys=keys, reference_state=reference_state) for cid in cids]
    V = torch.stack(vecs, dim=0)
    D = torch.cdist(V, V, p=2.0)
    distance = str(distance).lower()
    if distance in {"squared_l2", "squared", "l2_squared"}:
        D = D.pow(2)
    elif distance in {"l2", "euclidean"}:
        pass
    else:
        raise ValueError("distance must be 'squared_l2' or 'l2'.")

    scores = {}
    for i, cid in enumerate(cids):
        row = D[i].clone()
        row[i] = float("inf")  # exclude self only; keep zero distances to duplicate client updates
        nearest = torch.topk(row, k=nb, largest=False).values
        scores[cid] = float(nearest.sum().item())
    return scores


def krum_aggregate(
    submissions: Submissions,
    *,
    f: int = 1,
    reference_state: Optional[StateDict] = None,
    keys: Optional[Sequence[str]] = None,
    distance: str = "squared_l2",
    return_selected: bool = False,
):
    _check_submissions(submissions)
    n = len(submissions)
    if n <= 2 * int(f) + 2:
        out = coordinate_median_aggregate(submissions, keys=keys)
        return (out, []) if return_selected else out

    f = _safe_f_for_krum(n, f)
    scores = krum_scores(submissions, f=f, reference_state=reference_state, keys=keys, distance=distance)
    chosen = min(scores, key=scores.get)
    out = copy.deepcopy(submissions[chosen])
    return (out, [chosen]) if return_selected else out


def multikrum_aggregate(
    submissions: Submissions,
    *,
    f: int = 1,
    m: int = 0,
    reference_state: Optional[StateDict] = None,
    keys: Optional[Sequence[str]] = None,
    distance: str = "squared_l2",
    return_selected: bool = False,
):
    _check_submissions(submissions)
    n = len(submissions)
    if n <= 2 * int(f) + 2:
        out = coordinate_median_aggregate(submissions, keys=keys)
        return (out, []) if return_selected else out

    f = _safe_f_for_krum(n, f)
    max_m = max(1, n - f - 2)
    if m <= 0:
        m = max_m
    m = max(1, min(int(m), max_m))

    scores = krum_scores(submissions, f=f, reference_state=reference_state, keys=keys, distance=distance)
    selected = sorted(scores, key=scores.get)[:m]
    out = fedavg_aggregate({cid: submissions[cid] for cid in selected}, keys=keys)
    return (out, selected) if return_selected else out


def bulyan_aggregate(
    submissions: Submissions,
    *,
    f: int = 1,
    reference_state: Optional[StateDict] = None,
    keys: Optional[Sequence[str]] = None,
    distance: str = "squared_l2",
    return_selected: bool = False,
):
    """Formal Bulyan: iterative Krum preselection, then median-centered coordinate filtering."""
    _check_submissions(submissions)
    n = len(submissions)
    if keys is None:
        keys = _float_keys(next(iter(submissions.values())))

    f = int(max(0, f))
    if n < 4 * f + 3 or f == 0:
        if f == 0:
            out = fedavg_aggregate(submissions, keys=keys)
        else:
            out = trimmed_mean_aggregate(submissions, f=f, keys=keys)
        return (out, []) if return_selected else out

    theta = n - 2 * f
    beta = max(1, theta - 2 * f)  # n - 4f

    remaining = {cid: copy.deepcopy(st) for cid, st in submissions.items()}
    selected: List[int] = []

    for _ in range(theta):
        scores = krum_scores(remaining, f=f, reference_state=reference_state, keys=keys, distance=distance)
        chosen = min(scores, key=scores.get)
        selected.append(chosen)
        remaining.pop(chosen)

    selected_submissions = {cid: submissions[cid] for cid in selected}
    template = _copy_template(selected_submissions)

    for k in keys:
        stack = torch.stack([selected_submissions[cid][k].detach().float().cpu() for cid in selected], dim=0)
        med = torch.median(stack, dim=0).values
        dist_to_med = torch.abs(stack - med.unsqueeze(0))
        k_keep = min(beta, stack.shape[0])
        closest_idx = torch.topk(dist_to_med, k=k_keep, dim=0, largest=False).indices
        closest_vals = torch.gather(stack, dim=0, index=closest_idx)
        template[k] = closest_vals.mean(dim=0).to(dtype=template[k].dtype)

    return (template, selected) if return_selected else template


def robust_aggregate(
    method: str,
    submissions: Submissions,
    args,
    *,
    reference_state: Optional[StateDict] = None,
    return_selected: bool = False,
):
    """
    Unified robust aggregation entry point.

    Pass reference_state=global_state when submissions are full client model states.
    Then Krum/Multi-Krum/Bulyan score distances between client deltas instead of raw states.
    """
    _check_submissions(submissions)
    method = str(method).lower()
    n = len(submissions)

    f = int(getattr(args, "robust_f", -1))
    if f < 0:
        f = int(math.floor(float(getattr(args, "malicious_ratio", 0.0)) * n))
    f = max(0, min(f, max(0, n - 1)))

    keys = _float_keys(next(iter(submissions.values())))
    distance = str(getattr(args, "krum_distance", "squared_l2")).lower()

    if method in {"fedavg", "avg", "mean"}:
        out = fedavg_aggregate(submissions, keys=keys)
        return (out, sorted(submissions.keys())) if return_selected else out

    if method in {"median", "coordinate_median", "coordinate-median"}:
        out = coordinate_median_aggregate(submissions, keys=keys)
        return (out, sorted(submissions.keys())) if return_selected else out

    if method in {"trimmed_mean", "trimmed-mean", "trimmean"}:
        trim_count = int(getattr(args, "trim_count", -1))
        if trim_count < 0:
            trim_count = f
        out = trimmed_mean_aggregate(submissions, f=trim_count, keys=keys)
        return (out, sorted(submissions.keys())) if return_selected else out

    if method == "krum":
        return krum_aggregate(
            submissions, f=f, reference_state=reference_state, keys=keys, distance=distance, return_selected=return_selected
        )

    if method in {"multikrum", "multi_krum", "multi-krum"}:
        return multikrum_aggregate(
            submissions,
            f=f,
            m=int(getattr(args, "multikrum_m", 0)),
            reference_state=reference_state,
            keys=keys,
            distance=distance,
            return_selected=return_selected,
        )

    if method == "bulyan":
        return bulyan_aggregate(
            submissions, f=f, reference_state=reference_state, keys=keys, distance=distance, return_selected=return_selected
        )

    raise ValueError(f"Unknown robust aggregation method: {method}")
