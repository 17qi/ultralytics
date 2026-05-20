from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.torch_utils import get_flops, get_num_params


@dataclass
class RoiCounters:
    total: int = 0
    hit: int = 0

    @property
    def recall(self) -> float:
        return float(self.hit) / float(self.total) if self.total > 0 else 0.0


def _derive_experiment(weights_path: Path) -> str:
    parts = list(weights_path.parts)
    if "weights" in parts:
        weights_idx = parts.index("weights")
        if weights_idx >= 1:
            return parts[weights_idx - 1]
    return weights_path.stem


class RoiRecallValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None, roi_conf=0.25, roi_iou=0.5):
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.roi_conf = float(roi_conf)
        self.roi_iou = float(roi_iou)
        self.roi = RoiCounters()

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)
            gt_cls = pbatch["cls"]
            gt_boxes = pbatch["bboxes"]
            if gt_boxes.numel() == 0:
                continue
            self.roi.total += int(gt_boxes.shape[0])

            if predn["bboxes"].numel() == 0:
                continue

            conf_mask = predn["conf"] >= self.roi_conf
            if conf_mask.sum() == 0:
                continue
            pred_boxes = predn["bboxes"][conf_mask]
            pred_cls = predn["cls"][conf_mask]

            hits = 0
            for j in range(gt_boxes.shape[0]):
                cls_mask = pred_cls == gt_cls[j]
                if cls_mask.sum() == 0:
                    continue
                ious = box_iou(gt_boxes[j : j + 1], pred_boxes[cls_mask]).squeeze(0)
                if ious.numel() and float(ious.max()) >= self.roi_iou:
                    hits += 1
            self.roi.hit += hits


def _get_fps(speed: dict[str, float]) -> float:
    total = float(speed.get("preprocess", 0.0)) + float(speed.get("inference", 0.0)) + float(
        speed.get("postprocess", 0.0)
    )
    if total > 0:
        return 1000.0 / total
    inf = float(speed.get("inference", 0.0))
    return 1000.0 / inf if inf > 0 else 0.0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    weights = r"C:\git_lib\YOLO_test\YOLO11\runs\detect\train64\weights\best.pt"
    data = r"data_3_yolo_mix\weed_lettuce.yaml"
    split = "val"
    imgsz = 640
    metric_conf = 0.001
    metric_iou = 0.7
    roi_conf = 0.25
    roi_iou = 0.5
    device = "0"
    batch = 4
    workers = 0
    experiment = ""
    summary = ""

    weights_path = Path(weights)
    eval_dir = weights_path.parent / "roi_eval"
    experiment = experiment or _derive_experiment(weights_path)

    model = YOLO(str(weights_path))
    params_m = get_num_params(model.model) / 1e6
    try:
        gflops = get_flops(model.model, imgsz=imgsz)
    except Exception:
        gflops = "NA"

    metric_validator = DetectionValidator(
        save_dir=eval_dir,
        args={
            "model": str(weights_path),
            "data": data,
            "split": split,
            "imgsz": imgsz,
            "conf": metric_conf,
            "iou": metric_iou,
            "device": device,
            "batch": batch,
            "workers": workers,
        },
    )
    stats = metric_validator()

    speed = metric_validator.speed
    fps = _get_fps(speed)

    roi_validator = RoiRecallValidator(
        save_dir=eval_dir,
        args={
            "model": str(weights_path),
            "data": data,
            "split": split,
            "imgsz": imgsz,
            "conf": roi_conf,
            "iou": roi_iou,
            "device": device,
            "batch": batch,
            "workers": workers,
        },
        roi_conf=roi_conf,
        roi_iou=roi_iou,
    )
    roi_validator(model=model.model)

    precision = _safe_float(stats.get("metrics/precision(B)"))
    recall = _safe_float(stats.get("metrics/recall(B)"))
    map50 = _safe_float(stats.get("metrics/mAP50(B)"))
    map50_95 = _safe_float(stats.get("metrics/mAP50-95(B)"))

    model_yaml = str(getattr(model.model, "yaml_file", "") or "")
    model_name = model_yaml or model.model.__class__.__name__
    row = {
        "experiment": experiment,
        "model": model_name,
        "weights": str(weights_path),
        "data": data,
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "metric_conf": metric_conf,
        "metric_iou": metric_iou,
        "roi_conf": roi_conf,
        "roi_iou": roi_iou,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "params_m": params_m,
        "gflops": gflops,
        "fps": fps,
        "roi_recall": roi_validator.roi.recall,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    summary_path = Path(summary) if summary else eval_dir / "model_performance_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment",
            "model",
            "weights",
            "data",
            "split",
            "imgsz",
            "batch",
            "metric_conf",
            "metric_iou",
            "roi_conf",
            "roi_iou",
            "precision",
            "recall",
            "map50",
            "map50_95",
            "params_m",
            "gflops",
            "fps",
            "roi_recall",
            "timestamp",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Experiment: {experiment}")
    print(f"Metric conf/iou: {metric_conf} / {metric_iou}")
    print(f"ROI conf/iou: {roi_conf} / {roi_iou}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"mAP@0.5: {map50}")
    print(f"mAP@0.5:0.95: {map50_95}")
    print(f"Params(M): {params_m}")
    print(f"GFLOPs: {gflops}")
    print(f"FPS: {fps}")
    print(f"ROI Recall: {roi_validator.roi.recall}")
    print(f"Saved to: {summary_path}")


if __name__ == "__main__":
    main()
