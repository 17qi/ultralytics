from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_rgbd_dataset import run_audit


def _write_rgb(path: Path, size: tuple[int, int]) -> None:
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    Image.fromarray(image).save(path)


def _write_depth(path: Path, size: tuple[int, int]) -> None:
    height, width = size
    depth = np.zeros((height, width), dtype=np.uint16)
    depth[:, :] = 1000
    Image.fromarray(depth, mode="I;16").save(path)


def test_audit_outputs(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "image"
    depth_dir = tmp_path / "depth_aligned"
    (rgb_dir / "train").mkdir(parents=True)
    (depth_dir / "train").mkdir(parents=True)

    _write_rgb(rgb_dir / "train" / "0001.png", (10, 12))
    _write_depth(depth_dir / "train" / "0001.png", (10, 12))

    reports_dir = tmp_path / "reports"
    pairs, overlays = run_audit(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        reports_dir=reports_dir,
        overlay_count=1,
        seed=1,
        invalid_threshold=0.99,
    )

    assert pairs == 1
    assert overlays == 1
    assert (reports_dir / "pair_check.csv").exists()
    assert (reports_dir / "depth_quality_summary.csv").exists()
    assert (reports_dir / "overlay_samples").is_dir()
    assert any((reports_dir / "overlay_samples").iterdir())
