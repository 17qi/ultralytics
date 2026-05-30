import argparse
import csv
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from matplotlib import cm


@dataclass
class PairRecord:
    key: str
    rgb_path: Optional[Path]
    depth_path: Optional[Path]
    paired: bool
    rgb_shape: str
    depth_shape: str
    rgb_mtime: Optional[float]
    depth_mtime: Optional[float]
    mtime_delta_sec: Optional[float]
    note: str


@dataclass
class DepthImageStats:
    key: str
    split: str
    depth_path: Path
    width: int
    height: int
    depth_dtype: str
    depth_zero_ratio: float


def _iter_pngs(root: Path) -> Iterable[Path]:
    return root.rglob("*.png")


def _rel_key(path: Path, root: Path, stem_only: bool) -> str:
    if stem_only:
        return path.stem
    rel = path.relative_to(root)
    return str(rel.with_suffix(""))


def _read_rgb(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return np.array(rgb)
    except OSError:
        return None


def _read_depth(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as image:
            depth = np.array(image)
    except OSError:
        return None
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def _shape_str(image: Optional[np.ndarray]) -> str:
    if image is None:
        return ""
    if image.ndim == 2:
        h, w = image.shape
        return f"{h}x{w}"
    h, w, c = image.shape
    return f"{h}x{w}x{c}"


def _mtime(path: Optional[Path]) -> Optional[float]:
    if path is None:
        return None
    return path.stat().st_mtime


def _format_time(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _normalize_depth_to_u8(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=np.uint8)
    values = depth[valid].astype(np.float32)
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (depth.astype(np.float32) - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _blend_overlay(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    depth_u8 = _normalize_depth_to_u8(depth)
    cmap = cm.get_cmap("turbo", 256)
    lut = (cmap(np.arange(256))[:, :3] * 255.0).astype(np.uint8)
    depth_color = lut[depth_u8]
    alpha = 0.6
    blended = rgb.astype(np.float32) * (1.0 - alpha) + depth_color.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _has_root_pngs(root: Path) -> bool:
    return any(root.glob("*.png"))


def _collect_pairs(rgb_dir: Path, depth_dir: Path) -> Tuple[List[PairRecord], Dict[str, Path], Dict[str, Path]]:
    rgb_paths = list(_iter_pngs(rgb_dir))
    depth_paths = list(_iter_pngs(depth_dir))

    depth_stem_only = _has_root_pngs(depth_dir)
    rgb_stem_only = depth_stem_only

    rgb_map = {_rel_key(p, rgb_dir, rgb_stem_only): p for p in rgb_paths}
    depth_map = {_rel_key(p, depth_dir, depth_stem_only): p for p in depth_paths}

    keys = sorted(set(rgb_map.keys()) | set(depth_map.keys()))
    records: List[PairRecord] = []
    for key in keys:
        rgb_path = rgb_map.get(key)
        depth_path = depth_map.get(key)
        paired = rgb_path is not None and depth_path is not None
        rgb_image = _read_rgb(rgb_path) if rgb_path else None
        depth_image = _read_depth(depth_path) if depth_path else None
        rgb_shape = _shape_str(rgb_image)
        depth_shape = _shape_str(depth_image)
        rgb_time = _mtime(rgb_path)
        depth_time = _mtime(depth_path)
        mtime_delta = None
        if rgb_time is not None and depth_time is not None:
            mtime_delta = abs(rgb_time - depth_time)
        note = ""
        if not paired:
            note = "missing_rgb" if rgb_path is None else "missing_depth"
        records.append(
            PairRecord(
                key=key,
                rgb_path=rgb_path,
                depth_path=depth_path,
                paired=paired,
                rgb_shape=rgb_shape,
                depth_shape=depth_shape,
                rgb_mtime=rgb_time,
                depth_mtime=depth_time,
                mtime_delta_sec=mtime_delta,
                note=note,
            )
        )
    return records, rgb_map, depth_map


def _split_from_key(key: str) -> str:
    parts = Path(key).parts
    if len(parts) >= 2:
        return parts[0]
    return ""


def _analyze_depth_images(depth_map: Dict[str, Path]) -> Tuple[List[DepthImageStats], List[float], int]:
    stats: List[DepthImageStats] = []
    col_invalid_counts: List[int] = []
    col_total_counts: List[int] = []
    max_width = 0

    for key, depth_path in sorted(depth_map.items()):
        depth = _read_depth(depth_path)
        if depth is None:
            continue
        if depth.ndim != 2:
            depth = depth[:, :, 0]
        height, width = depth.shape
        max_width = max(max_width, width)
        if len(col_invalid_counts) < width:
            extend = width - len(col_invalid_counts)
            col_invalid_counts.extend([0] * extend)
            col_total_counts.extend([0] * extend)
        invalid_mask = depth == 0
        col_invalid_counts[:width] = [
            col_invalid_counts[i] + int(invalid_mask[:, i].sum()) for i in range(width)
        ]
        col_total_counts[:width] = [col_total_counts[i] + height for i in range(width)]

        depth_zero_ratio = float(invalid_mask.mean())
        stats.append(
            DepthImageStats(
                key=key,
                split=_split_from_key(key),
                depth_path=depth_path,
                width=width,
                height=height,
                depth_dtype=str(depth.dtype),
                depth_zero_ratio=depth_zero_ratio,
            )
        )

    col_invalid_ratio: List[float] = []
    for invalid, total in zip(col_invalid_counts, col_total_counts):
        if total == 0:
            col_invalid_ratio.append(0.0)
        else:
            col_invalid_ratio.append(float(invalid) / float(total))

    return stats, col_invalid_ratio, max_width


def _estimate_invalid_edges(col_invalid_ratio: Sequence[float], threshold: float) -> Tuple[int, int]:
    left_end = -1
    for idx, ratio in enumerate(col_invalid_ratio):
        if ratio >= threshold:
            left_end = idx
        else:
            break

    right_start = -1
    for idx in range(len(col_invalid_ratio) - 1, -1, -1):
        if col_invalid_ratio[idx] >= threshold:
            right_start = idx
        else:
            break

    return left_end, right_start


def _write_pair_check_csv(records: Sequence[PairRecord], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "key",
                "rgb_path",
                "depth_path",
                "paired",
                "rgb_shape",
                "depth_shape",
                "rgb_mtime",
                "depth_mtime",
                "mtime_delta_sec",
                "note",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.key,
                    str(record.rgb_path) if record.rgb_path else "",
                    str(record.depth_path) if record.depth_path else "",
                    int(record.paired),
                    record.rgb_shape,
                    record.depth_shape,
                    _format_time(record.rgb_mtime),
                    _format_time(record.depth_mtime),
                    f"{record.mtime_delta_sec:.3f}" if record.mtime_delta_sec is not None else "",
                    record.note,
                ]
            )


def _write_depth_quality_csv(
    stats: Sequence[DepthImageStats],
    col_invalid_ratio: Sequence[float],
    threshold: float,
    output_path: Path,
) -> None:
    left_end, right_start = _estimate_invalid_edges(col_invalid_ratio, threshold)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "record_type",
                "key",
                "split",
                "depth_path",
                "width",
                "height",
                "depth_dtype",
                "depth_zero_ratio",
                "column_index",
                "column_invalid_ratio",
                "left_invalid_end",
                "right_invalid_start",
                "invalid_threshold",
            ]
        )
        for item in stats:
            writer.writerow(
                [
                    "image",
                    item.key,
                    item.split,
                    str(item.depth_path),
                    item.width,
                    item.height,
                    item.depth_dtype,
                    f"{item.depth_zero_ratio:.6f}",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
        for idx, ratio in enumerate(col_invalid_ratio):
            writer.writerow(
                [
                    "column",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    idx,
                    f"{ratio:.6f}",
                    "",
                    "",
                    "",
                ]
            )
        writer.writerow(
            [
                "summary",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                left_end,
                right_start,
                f"{threshold:.3f}",
            ]
        )


def _write_overlays(
    pairs: Sequence[PairRecord],
    output_dir: Path,
    overlay_count: int,
    seed: int,
) -> int:
    paired = [record for record in pairs if record.paired]
    if not paired:
        return 0
    random.seed(seed)
    sample_count = min(overlay_count, len(paired))
    sampled = random.sample(paired, sample_count)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for record in sampled:
        if record.rgb_path is None or record.depth_path is None:
            continue
        rgb = _read_rgb(record.rgb_path)
        depth = _read_depth(record.depth_path)
        if rgb is None or depth is None:
            continue
        if rgb.shape[:2] != depth.shape[:2]:
            continue
        overlay = _blend_overlay(rgb, depth)
        filename = record.key.replace(os.sep, "_") + "_overlay.png"
        out_path = output_dir / filename
        Image.fromarray(overlay).save(out_path)
        written += 1
    return written


def run_audit(
    rgb_dir: Path,
    depth_dir: Path,
    reports_dir: Path,
    overlay_count: int,
    seed: int,
    invalid_threshold: float,
) -> Tuple[int, int]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = reports_dir / "overlay_samples"

    pairs, _, depth_map = _collect_pairs(rgb_dir, depth_dir)
    pair_csv = reports_dir / "pair_check.csv"
    _write_pair_check_csv(pairs, pair_csv)

    stats, col_invalid_ratio, _ = _analyze_depth_images(depth_map)
    depth_csv = reports_dir / "depth_quality_summary.csv"
    _write_depth_quality_csv(stats, col_invalid_ratio, invalid_threshold, depth_csv)

    overlay_written = _write_overlays(pairs, overlay_dir, overlay_count, seed)
    return len(pairs), overlay_written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RGB-D dataset without modifying data.")
    parser.add_argument("--plan", type=Path, default=Path("doc/data-processing-plan.md"))
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--overlay-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invalid-threshold", type=float, default=0.99)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {args.rgb_dir}")
    if not args.depth_dir.exists():
        raise FileNotFoundError(f"Depth directory not found: {args.depth_dir}")
    if args.plan.exists():
        _ = args.plan.read_text(encoding="utf-8")

    total_pairs, overlays = run_audit(
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        reports_dir=args.reports_dir,
        overlay_count=args.overlay_count,
        seed=args.seed,
        invalid_threshold=args.invalid_threshold,
    )
    print(f"[INFO] pairs={total_pairs}, overlays={overlays}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
