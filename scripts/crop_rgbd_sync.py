from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml


DEFAULT_CONFIG_PATH = Path("doc/crop_config.yaml")
DEFAULT_RGB_ROOT = Path("C:\\Users\\14312\\Desktop\\rgbd_data_mass_2\\images")
DEFAULT_DEPTH_ROOT = Path("C:\\Users\\14312\\Desktop\\rgbd_data_mass_2\\depth_aligned")
DEFAULT_OUTPUT_ROOT = Path("C:\\Users\\14312\\Desktop\\rgbd_data_mass_2\\data_processed")
DEFAULT_REPORTS_DIR = Path("C:\\Users\\14312\\Desktop\\rgbd_data_mass_2\\data_processed\\reports")
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_FLAT_SPLIT = "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop RGB and aligned depth with a shared ROI.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to crop_config.yaml.")
    parser.add_argument("--rgb-dir", default=str(DEFAULT_RGB_ROOT), help="RGB root (flat or split).")
    parser.add_argument("--depth-dir", default=str(DEFAULT_DEPTH_ROOT), help="Depth root (flat or split).")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="Reports directory.")
    parser.add_argument(
        "--flat-split",
        default=DEFAULT_FLAT_SPLIT,
        help="If RGB root is flat, use this split name for output.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan and report without writing files.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def validate_config(cfg: dict) -> tuple[int, int, int, int, int, int]:
    source_width = int(cfg.get("source_width"))
    source_height = int(cfg.get("source_height"))
    left_crop = int(cfg.get("left_crop"))
    right_crop = int(cfg.get("right_crop"))
    top_crop = int(cfg.get("top_crop"))
    bottom_crop = int(cfg.get("bottom_crop"))
    output_width = int(cfg.get("output_width"))
    output_height = int(cfg.get("output_height"))

    if any(v < 0 for v in (left_crop, right_crop, top_crop, bottom_crop)):
        raise ValueError("Crop values must be non-negative.")
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source_width and source_height must be positive.")
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output_width and output_height must be positive.")

    expected_width = source_width - left_crop - right_crop
    expected_height = source_height - top_crop - bottom_crop
    if expected_width != output_width or expected_height != output_height:
        raise ValueError(
            "Crop config mismatch: output size does not match source size minus crops."
        )

    return source_width, source_height, left_crop, right_crop, top_crop, bottom_crop


def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def log_error(errors: list[str], message: str, dry_run: bool) -> None:
    errors.append(message)
    print(f"[ERROR] {message}")
    if dry_run:
        return


def pick_depth_vis_gray_path(depth_vis_dir: Path, stem: str) -> Path | None:
    for suffix in ("_gray_u8.png", "_gray_u16.png"):
        candidate = depth_vis_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def process_split(
    split: str,
    rgb_root: Path,
    depth_root: Path,
    depth_is_flat: bool,
    depth_names_all: set[str] | None,
    depth_vis_root: Path | None,
    depth_vis_is_flat: bool,
    valid_mask_root: Path | None,
    valid_mask_is_flat: bool,
    out_root: Path,
    source_width: int,
    source_height: int,
    left_crop: int,
    right_crop: int,
    top_crop: int,
    bottom_crop: int,
    dry_run: bool,
    errors: list[str],
) -> tuple[int, int]:
    # Prepare input/output directories for this split.
    rgb_dir = rgb_root / split
    depth_dir = depth_root if depth_is_flat else (depth_root / split)
    depth_vis_dir = None
    if depth_vis_root is not None:
        depth_vis_dir = depth_vis_root if depth_vis_is_flat else (depth_vis_root / split)
    valid_mask_dir = None
    if valid_mask_root is not None:
        valid_mask_dir = valid_mask_root if valid_mask_is_flat else (valid_mask_root / split)
    out_rgb_dir = out_root / "images" / split
    out_depth_dir = out_root / "depth_aligned" / split
    out_depth_vis_dir = out_root / "depth_vis_aligned" / split
    out_valid_mask_dir = out_root / "depth_aligned_valid_mask" / split

    if not rgb_dir.is_dir():
        log_error(errors, f"Missing RGB split dir: {rgb_dir}", dry_run)
        return 0, 0
    if not depth_is_flat and not depth_dir.is_dir():
        log_error(errors, f"Missing depth split dir: {depth_dir}", dry_run)
        return 0, 0
    if depth_vis_dir is not None and not depth_vis_is_flat and not depth_vis_dir.is_dir():
        log_error(errors, f"Missing depth_vis split dir: {depth_vis_dir}", dry_run)
        return 0, 0
    if valid_mask_dir is not None and not valid_mask_is_flat and not valid_mask_dir.is_dir():
        log_error(errors, f"Missing valid_mask split dir: {valid_mask_dir}", dry_run)
        return 0, 0

    ensure_dir(out_rgb_dir, dry_run)
    ensure_dir(out_depth_dir, dry_run)
    if depth_vis_dir is not None:
        ensure_dir(out_depth_vis_dir, dry_run)
    if valid_mask_dir is not None:
        ensure_dir(out_valid_mask_dir, dry_run)

    rgb_files = sorted(rgb_dir.glob("*.png"))
    if not depth_is_flat:
        depth_files = sorted(depth_dir.glob("*.png"))
        rgb_names = {p.name for p in rgb_files}
        depth_names = {p.name for p in depth_files}

        for extra in sorted(depth_names - rgb_names):
            log_error(errors, f"Depth without RGB ({split}): {extra}", dry_run)
    else:
        depth_files = []

    processed = 0
    skipped = 0
    for rgb_path in rgb_files:
        # Load core RGB/Depth pair first.
        depth_path = depth_dir / rgb_path.name
        if not depth_path.exists():
            log_error(errors, f"Missing depth ({split}): {rgb_path.name}", dry_run)
            skipped += 1
            continue

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            log_error(errors, f"Failed to read RGB ({split}): {rgb_path}", dry_run)
            skipped += 1
            continue

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            log_error(errors, f"Failed to read depth ({split}): {depth_path}", dry_run)
            skipped += 1
            continue

        if rgb.shape[:2] != depth.shape[:2]:
            log_error(
                errors,
                f"Size mismatch ({split}): {rgb_path.name} rgb={rgb.shape[:2]} depth={depth.shape[:2]}",
                dry_run,
            )
            skipped += 1
            continue

        if rgb.shape[1] != source_width or rgb.shape[0] != source_height:
            log_error(
                errors,
                f"Source size mismatch ({split}): {rgb_path.name} size={rgb.shape[1]}x{rgb.shape[0]}",
                dry_run,
            )
            skipped += 1
            continue

        x1 = left_crop
        x2 = rgb.shape[1] - right_crop
        y1 = top_crop
        y2 = rgb.shape[0] - bottom_crop
        if x2 <= x1 or y2 <= y1:
            log_error(errors, f"Invalid crop window ({split}): {rgb_path.name}", dry_run)
            skipped += 1
            continue

        rgb_crop = rgb[y1:y2, x1:x2]
        depth_crop = depth[y1:y2, x1:x2]
        depth_vis_gray = None
        depth_vis_color = None
        valid_mask = None

        if depth_vis_dir is not None:
            stem = rgb_path.stem
            depth_vis_gray_path = pick_depth_vis_gray_path(depth_vis_dir, stem)
            depth_vis_color_path = depth_vis_dir / f"{stem}_colormap.png"
            if depth_vis_gray_path is None:
                log_error(errors, f"Missing depth_vis gray ({split}): {stem}_gray_u8.png|{stem}_gray_u16.png", dry_run)
            if not depth_vis_color_path.exists():
                log_error(errors, f"Missing depth_vis colormap ({split}): {depth_vis_color_path.name}", dry_run)
            if depth_vis_gray_path is not None:
                depth_vis_gray = cv2.imread(str(depth_vis_gray_path), cv2.IMREAD_UNCHANGED)
                if depth_vis_gray is None:
                    log_error(errors, f"Failed to read depth_vis gray ({split}): {depth_vis_gray_path}", dry_run)
            if depth_vis_color_path.exists():
                depth_vis_color = cv2.imread(str(depth_vis_color_path), cv2.IMREAD_UNCHANGED)
                if depth_vis_color is None:
                    log_error(errors, f"Failed to read depth_vis colormap ({split}): {depth_vis_color_path}", dry_run)

        if valid_mask_dir is not None:
            valid_mask_path = valid_mask_dir / rgb_path.name
            if not valid_mask_path.exists():
                log_error(errors, f"Missing valid_mask ({split}): {valid_mask_path.name}", dry_run)
            else:
                valid_mask = cv2.imread(str(valid_mask_path), cv2.IMREAD_UNCHANGED)
                if valid_mask is None:
                    log_error(errors, f"Failed to read valid_mask ({split}): {valid_mask_path}", dry_run)

        if not dry_run:
            out_rgb = out_rgb_dir / rgb_path.name
            out_depth = out_depth_dir / depth_path.name
            if not cv2.imwrite(str(out_rgb), rgb_crop):
                log_error(errors, f"Failed to write RGB ({split}): {out_rgb}", dry_run)
                skipped += 1
                continue
            if not cv2.imwrite(str(out_depth), depth_crop):
                log_error(errors, f"Failed to write depth ({split}): {out_depth}", dry_run)
                skipped += 1
                continue
            if depth_vis_dir is not None and depth_vis_gray is not None:
                gray_suffix = "_gray_u8.png" if depth_vis_gray.dtype == "uint8" else "_gray_u16.png"
                out_gray = out_depth_vis_dir / f"{rgb_path.stem}{gray_suffix}"
                if not cv2.imwrite(str(out_gray), depth_vis_gray[y1:y2, x1:x2]):
                    log_error(errors, f"Failed to write depth_vis gray ({split}): {out_gray}", dry_run)
            if depth_vis_dir is not None and depth_vis_color is not None:
                out_color = out_depth_vis_dir / f"{rgb_path.stem}_colormap.png"
                if not cv2.imwrite(str(out_color), depth_vis_color[y1:y2, x1:x2]):
                    log_error(errors, f"Failed to write depth_vis colormap ({split}): {out_color}", dry_run)
            if valid_mask_dir is not None and valid_mask is not None:
                out_mask = out_valid_mask_dir / rgb_path.name
                if not cv2.imwrite(str(out_mask), valid_mask[y1:y2, x1:x2]):
                    log_error(errors, f"Failed to write valid_mask ({split}): {out_mask}", dry_run)

        processed += 1

    return processed, skipped


def write_error_log(errors: list[str], reports_dir: Path, dry_run: bool) -> None:
    if dry_run or not errors:
        return
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_dir / "crop_errors.log"
    with log_path.open("w", encoding="utf-8") as handle:
        for line in errors:
            handle.write(f"{line}\n")
    print(f"[INFO] Wrote error log: {log_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    source_width, source_height, left_crop, right_crop, top_crop, bottom_crop = validate_config(cfg)

    rgb_root = Path(args.rgb_dir)
    depth_root = Path(args.depth_dir)
    out_root = Path(args.output_root)
    reports_dir = Path(args.reports_dir)
    dry_run = bool(args.dry_run)

    errors: list[str] = []
    total_processed = 0
    total_skipped = 0
    rgb_names_all: set[str] = set()

    rgb_split_dirs = [rgb_root / split for split in DEFAULT_SPLITS if (rgb_root / split).is_dir()]
    rgb_is_flat = not rgb_split_dirs and bool(list(rgb_root.glob("*.png")))
    splits = (args.flat_split,) if rgb_is_flat else DEFAULT_SPLITS
    if rgb_is_flat:
        log_error(errors, f"RGB root has no split dirs; using flat RGB dir: {rgb_root}", dry_run)

    depth_split_dirs = [depth_root / split for split in splits if (depth_root / split).is_dir()]
    depth_root_pngs = sorted(depth_root.glob("*.png"))
    depth_is_flat = not depth_split_dirs and bool(depth_root_pngs)
    depth_names_all: set[str] | None = {p.name for p in depth_root_pngs} if depth_is_flat else None
    if depth_is_flat:
        log_error(errors, f"Depth root has no split dirs; using flat depth dir: {depth_root}", dry_run)

    if rgb_is_flat and depth_split_dirs:
        log_error(errors, "RGB is flat but depth has split dirs; using flat depth root for pairing.", dry_run)

    base_root = rgb_root.parent
    depth_vis_root = base_root / "depth_vis_aligned"
    valid_mask_root = base_root / "depth_aligned_valid_mask"
    depth_vis_is_flat = False
    valid_mask_is_flat = False
    if depth_vis_root.is_dir():
        depth_vis_split_dirs = [depth_vis_root / split for split in splits if (depth_vis_root / split).is_dir()]
        depth_vis_is_flat = not depth_vis_split_dirs and bool(list(depth_vis_root.glob("*.png")))
        if depth_vis_is_flat:
            log_error(
                errors,
                f"Depth vis root has no split dirs; using flat depth_vis dir: {depth_vis_root}",
                dry_run,
            )
    else:
        depth_vis_root = None

    if valid_mask_root.is_dir():
        valid_mask_split_dirs = [valid_mask_root / split for split in splits if (valid_mask_root / split).is_dir()]
        valid_mask_is_flat = not valid_mask_split_dirs and bool(list(valid_mask_root.glob("*.png")))
        if valid_mask_is_flat:
            log_error(
                errors,
                f"Valid mask root has no split dirs; using flat valid_mask dir: {valid_mask_root}",
                dry_run,
            )
    else:
        valid_mask_root = None

    print("[INFO] Crop plan:")
    print(f"  source: {source_width}x{source_height}")
    print(f"  crop: left={left_crop} right={right_crop} top={top_crop} bottom={bottom_crop}")
    print(f"  rgb_root: {rgb_root}")
    print(f"  depth_root: {depth_root}")
    print(f"  output_root: {out_root}")
    print(f"  dry_run: {dry_run}")
    print(f"  depth_vis_root: {depth_vis_root}")
    print(f"  valid_mask_root: {valid_mask_root}")

    for split in splits:
        rgb_dir = rgb_root / split
        if rgb_is_flat:
            rgb_names_all.update({p.name for p in rgb_root.glob("*.png")})
        elif rgb_dir.is_dir():
            rgb_names_all.update({p.name for p in rgb_dir.glob("*.png")})
        processed, skipped = process_split(
            split,
            rgb_root,
            depth_root,
            depth_is_flat,
            depth_names_all,
            depth_vis_root,
            depth_vis_is_flat,
            valid_mask_root,
            valid_mask_is_flat,
            out_root,
            source_width,
            source_height,
            left_crop,
            right_crop,
            top_crop,
            bottom_crop,
            dry_run,
            errors,
        )
        total_processed += processed
        total_skipped += skipped

    if depth_is_flat and depth_names_all is not None:
        for extra in sorted(depth_names_all - rgb_names_all):
            log_error(errors, f"Depth without RGB (flat): {extra}", dry_run)

    print(f"[INFO] Done. processed={total_processed} skipped={total_skipped} errors={len(errors)}")
    write_error_log(errors, reports_dir, dry_run)


if __name__ == "__main__":
    main()
