import argparse
import os
from functools import lru_cache

import cv2
import numpy as np
import pyrealsense2 as rs


def get_config():
    args = argparse.Namespace(
        output="rgbd_data_1280x720",
        zmin=0.2,
        zmax=1.5,
        cmap="TURBO",
        save_filtered_depth=True,
        spatial_magnitude=2.0,
        spatial_alpha=0.5,
        spatial_delta=20.0,
        spatial_holes_fill=1.0,
        save_depth_meters=True,
        save_valid_mask=True,
        preview_scale=0.5,
    )
    if args.zmin >= args.zmax:
        raise ValueError("zmin must be smaller than zmax")
    return args


def ensure_dirs(base_dir: str, save_depth_meters: bool, save_valid_mask: bool, save_filtered_depth: bool):
    color_dir = os.path.join(base_dir, "color")
    depth_raw_dir = os.path.join(base_dir, "depth_raw")
    depth_aligned_dir = os.path.join(base_dir, "depth_aligned")
    depth_aligned_filtered_dir = os.path.join(base_dir, "depth_aligned_filtered")
    depth_vis_raw_dir = os.path.join(base_dir, "depth_vis_raw")
    depth_vis_aligned_dir = os.path.join(base_dir, "depth_vis_aligned")
    depth_vis_aligned_filtered_dir = os.path.join(base_dir, "depth_vis_aligned_filtered")

    dirs = [
        color_dir,
        depth_raw_dir,
        depth_aligned_dir,
        depth_vis_raw_dir,
        depth_vis_aligned_dir,
    ]
    if save_filtered_depth:
        dirs.append(depth_aligned_filtered_dir)
        dirs.append(depth_vis_aligned_filtered_dir)

    depth_m_dir = None
    if save_depth_meters:
        depth_m_dir = os.path.join(base_dir, "depth_meters")
        dirs.append(depth_m_dir)

    depth_valid_mask_dir = None
    if save_valid_mask:
        depth_valid_mask_dir = os.path.join(base_dir, "depth_aligned_valid_mask")
        dirs.append(depth_valid_mask_dir)

    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

    return {
        "color": color_dir,
        "depth_raw": depth_raw_dir,
        "depth_aligned": depth_aligned_dir,
        "depth_aligned_filtered": depth_aligned_filtered_dir if save_filtered_depth else None,
        "depth_vis_raw": depth_vis_raw_dir,
        "depth_vis_aligned": depth_vis_aligned_dir,
        "depth_vis_aligned_filtered": depth_vis_aligned_filtered_dir if save_filtered_depth else None,
        "depth_meters": depth_m_dir,
        "depth_valid_mask": depth_valid_mask_dir,
    }


def build_spatial_filter(args):
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, float(args.spatial_magnitude))
    spatial.set_option(rs.option.filter_smooth_alpha, float(args.spatial_alpha))
    spatial.set_option(rs.option.filter_smooth_delta, float(args.spatial_delta))
    spatial.set_option(rs.option.holes_fill, float(args.spatial_holes_fill))
    return spatial


def depth_to_gray_u16(depth_u16, depth_scale_m, zmin_m=0.2, zmax_m=1.5):
    depth_m = depth_u16.astype(np.float32) * float(depth_scale_m)
    depth_m = np.clip(depth_m, zmin_m, zmax_m)
    gray = ((depth_m - zmin_m) / (zmax_m - zmin_m) * 65535.0).astype(np.uint16)
    gray[depth_u16 == 0] = 0
    return gray


@lru_cache(maxsize=4)
def _build_colormap_lut_16(cmap_code: int):
    ramp_u8 = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    lut_256 = cv2.applyColorMap(ramp_u8, cmap_code).reshape(256, 3)
    idx_16 = (np.arange(65536, dtype=np.uint32) * 255 // 65535).astype(np.uint16)
    lut_16 = lut_256[idx_16]
    return lut_16


def _get_cmap_code(cmap_name: str):
    cmap_upper = cmap_name.upper()
    if cmap_upper == "TURBO":
        if hasattr(cv2, "COLORMAP_TURBO"):
            return cv2.COLORMAP_TURBO, "TURBO"
        return cv2.COLORMAP_JET, "JET"
    return cv2.COLORMAP_JET, "JET"


def depth_to_colormap_fixed(depth_u16, depth_scale_m, zmin_m=0.2, zmax_m=1.5, cmap="TURBO"):
    gray_u16 = depth_to_gray_u16(depth_u16, depth_scale_m, zmin_m=zmin_m, zmax_m=zmax_m)
    cmap_code, cmap_used = _get_cmap_code(cmap)
    lut_16 = _build_colormap_lut_16(int(cmap_code))
    depth_colormap = lut_16[gray_u16]
    depth_colormap[gray_u16 == 0] = 0
    return gray_u16, depth_colormap, cmap_used


def get_start_index(color_dir: str, depth_raw_dir: str):
    indices = []
    for directory in (color_dir, depth_raw_dir):
        for name in os.listdir(directory):
            stem, ext = os.path.splitext(name)
            if ext.lower() == ".png" and stem.isdigit():
                indices.append(int(stem))
    return (max(indices) + 1) if indices else 0


def safe_imwrite(path: str, image: np.ndarray) -> bool:
    try:
        ok = cv2.imwrite(path, image)
        if not ok:
            print(f"[ERROR] cv2.imwrite failed: {path}")
        return bool(ok)
    except Exception as exc:
        print(f"[ERROR] Exception in cv2.imwrite for {path}: {exc}")
        return False


def safe_save_npy(path: str, array: np.ndarray) -> bool:
    try:
        np.save(path, array)
        return True
    except Exception as exc:
        print(f"[ERROR] Exception in np.save for {path}: {exc}")
        return False


def main():
    args = get_config()

    dirs = ensure_dirs(args.output, args.save_depth_meters, args.save_valid_mask, args.save_filtered_depth)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # 将深度帧对齐到彩色相机坐标系（后续 aligned 数据与 color 像素一一对应）
    align = rs.align(rs.stream.color)
    spatial_filter = build_spatial_filter(args) if args.save_filtered_depth else None

    profile = pipeline.start(config)
    for _ in range(30):
        pipeline.wait_for_frames()

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    _, _, cmap_used = depth_to_colormap_fixed(
        np.zeros((1, 1), dtype=np.uint16), depth_scale, zmin_m=args.zmin, zmax_m=args.zmax, cmap=args.cmap
    )

    print(f"[INFO] depth_scale={depth_scale:.10f} m/unit")
    print(f"[INFO] vis_range=[{args.zmin:.3f}, {args.zmax:.3f}] m, cmap={cmap_used}")
    print(f"[INFO] save_depth_meters={args.save_depth_meters}, save_valid_mask={args.save_valid_mask}, save_filtered_depth={args.save_filtered_depth}")
    print(
        f"[INFO] spatial_filter: magnitude={args.spatial_magnitude}, alpha={args.spatial_alpha}, "
        f"delta={args.spatial_delta}, holes_fill={args.spatial_holes_fill}"
    )
    print("按空格保存一对 RGB/Depth，按 q 退出。")

    preview_window_name = "RGB | Depth (Fixed Range)"
    # 使用可缩放窗口，并将默认显示尺寸控制在笔记本更易查看的范围
    cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview_window_name, 1280, 420)

    counter = get_start_index(dirs["color"], dirs["depth_raw"])

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            # 原始深度（未对齐，来自 depth 传感器原生坐标系）
            depth_frame_raw = frames.get_depth_frame()
            # 对齐后深度（映射到 color 坐标系）
            depth_frame_aligned = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame_raw or not depth_frame_aligned or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_u16_raw = np.asanyarray(depth_frame_raw.get_data())
            depth_u16_aligned = np.asanyarray(depth_frame_aligned.get_data())
            depth_u16_aligned_filtered = None

            # 未对齐深度的可视化（仅保存，不用于预览窗口）
            depth_raw_gray_u16_save, depth_raw_colormap_save, _ = depth_to_colormap_fixed(
                depth_u16_raw, depth_scale, zmin_m=args.zmin, zmax_m=args.zmax, cmap=args.cmap
            )
            # 对齐后深度的可视化（用于预览窗口 + 保存）
            depth_gray_u16_save, depth_colormap_save, _ = depth_to_colormap_fixed(
                depth_u16_aligned, depth_scale, zmin_m=args.zmin, zmax_m=args.zmax, cmap=args.cmap
            )
            depth_aligned_filtered_gray_u16_save = None
            depth_aligned_filtered_colormap_save = None
            if args.save_filtered_depth and spatial_filter is not None:
                depth_frame_aligned_filtered = spatial_filter.process(depth_frame_aligned)
                depth_u16_aligned_filtered = np.asanyarray(depth_frame_aligned_filtered.get_data())
                depth_aligned_filtered_gray_u16_save, depth_aligned_filtered_colormap_save, _ = depth_to_colormap_fixed(
                    depth_u16_aligned_filtered, depth_scale, zmin_m=args.zmin, zmax_m=args.zmax, cmap=args.cmap
                )
            valid_mask = (depth_u16_aligned > 0).astype(np.uint8) * 255

            # 预览窗口右侧是 depth_colormap_save，即“对齐后深度图”的伪彩色结果
            preview = np.hstack((color_image, depth_colormap_save))
            # 仅缩放显示，不影响任何保存文件的分辨率
            if args.preview_scale != 1.0:
                preview_show = cv2.resize(
                    preview,
                    dsize=None,
                    fx=float(args.preview_scale),
                    fy=float(args.preview_scale),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                preview_show = preview
            cv2.imshow(preview_window_name, preview_show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                stem = f"{counter:04d}"
                color_path = os.path.join(dirs["color"], f"{stem}.png")
                depth_raw_path = os.path.join(dirs["depth_raw"], f"{stem}.png")
                depth_raw_gray_path = os.path.join(dirs["depth_vis_raw"], f"{stem}_gray_u16.png")
                depth_raw_colormap_path = os.path.join(dirs["depth_vis_raw"], f"{stem}_colormap.png")
                depth_aligned_path = os.path.join(dirs["depth_aligned"], f"{stem}.png")
                depth_gray_path = os.path.join(dirs["depth_vis_aligned"], f"{stem}_gray_u16.png")
                depth_colormap_path = os.path.join(dirs["depth_vis_aligned"], f"{stem}_colormap.png")
                depth_aligned_filtered_path = None
                depth_aligned_filtered_gray_path = None
                depth_aligned_filtered_colormap_path = None
                if args.save_filtered_depth and dirs["depth_aligned_filtered"] is not None:
                    depth_aligned_filtered_path = os.path.join(dirs["depth_aligned_filtered"], f"{stem}.png")
                    depth_aligned_filtered_gray_path = os.path.join(dirs["depth_vis_aligned_filtered"], f"{stem}_gray_u16.png")
                    depth_aligned_filtered_colormap_path = os.path.join(dirs["depth_vis_aligned_filtered"], f"{stem}_colormap.png")

                try:
                    key_ok = True
                    key_ok = safe_imwrite(color_path, color_image) and key_ok
                    key_ok = safe_imwrite(depth_raw_path, depth_u16_raw) and key_ok
                    key_ok = safe_imwrite(depth_raw_gray_path, depth_raw_gray_u16_save) and key_ok
                    key_ok = safe_imwrite(depth_raw_colormap_path, depth_raw_colormap_save) and key_ok
                    key_ok = safe_imwrite(depth_aligned_path, depth_u16_aligned) and key_ok
                    key_ok = safe_imwrite(depth_gray_path, depth_gray_u16_save) and key_ok
                    key_ok = safe_imwrite(depth_colormap_path, depth_colormap_save) and key_ok

                    if depth_aligned_filtered_path is not None and depth_u16_aligned_filtered is not None:
                        filtered_ok = True
                        filtered_ok = safe_imwrite(depth_aligned_filtered_path, depth_u16_aligned_filtered) and filtered_ok
                        filtered_ok = safe_imwrite(depth_aligned_filtered_gray_path, depth_aligned_filtered_gray_u16_save) and filtered_ok
                        filtered_ok = safe_imwrite(depth_aligned_filtered_colormap_path, depth_aligned_filtered_colormap_save) and filtered_ok
                        if filtered_ok:
                            print(f"[SAVED] depth_aligned_filtered(z16, spatial): {depth_aligned_filtered_path}")
                            print(f"[SAVED] depth_aligned_filtered_gray_u16: {depth_aligned_filtered_gray_path}")
                            print(f"[SAVED] depth_aligned_filtered_colormap: {depth_aligned_filtered_colormap_path}")
                        else:
                            print(f"[WARN] failed to save optional depth_aligned_filtered group for sample {stem}")

                    if args.save_depth_meters and dirs["depth_meters"] is not None:
                        depth_m_path = os.path.join(dirs["depth_meters"], f"{stem}.npy")
                        if safe_save_npy(depth_m_path, depth_u16_aligned.astype(np.float32) * depth_scale):
                            print(f"[SAVED] depth_meters: {depth_m_path}")
                        else:
                            print(f"[WARN] failed to save optional depth_meters: {depth_m_path}")

                    if args.save_valid_mask and dirs["depth_valid_mask"] is not None:
                        valid_mask_path = os.path.join(dirs["depth_valid_mask"], f"{stem}.png")
                        if safe_imwrite(valid_mask_path, valid_mask):
                            print(f"[SAVED] depth_valid_mask: {valid_mask_path}")
                        else:
                            print(f"[WARN] failed to save optional depth_valid_mask: {valid_mask_path}")

                    if key_ok:
                        print(f"[SAVED] color: {color_path}")
                        print(f"[SAVED] depth_raw(z16): {depth_raw_path}")
                        print(f"[SAVED] depth_raw_gray_u16: {depth_raw_gray_path}")
                        print(f"[SAVED] depth_raw_colormap: {depth_raw_colormap_path}")
                        print(f"[SAVED] depth_aligned(z16): {depth_aligned_path}")
                        print(f"[SAVED] depth_gray_u16: {depth_gray_path}")
                        print(f"[SAVED] depth_colormap({cmap_used}, zmin={args.zmin}, zmax={args.zmax}): {depth_colormap_path}")
                        counter += 1
                    else:
                        print(f"[ERROR] key files save failed for sample {stem}, not advancing counter")
                except Exception as exc:
                    print(f"[ERROR] Exception while saving sample {stem}: {exc}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

