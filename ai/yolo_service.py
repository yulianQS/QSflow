from __future__ import annotations

from pathlib import Path
from typing import Callable
import configparser
import math

from core.models import AnnotationBox, ImageItem, ImageStatus


class YoloService:
    def __init__(self) -> None:
        self.model = None
        self.model_path: Path | None = None
        self.custom_tflite = None
        self.input_size: tuple[int, int] | None = None
        self.model_description = ""

    def load(self, path: Path, class_names: list[str] | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("未安装 ultralytics，请执行 pip install -r requirements.txt") from exc
        supported = {".pt", ".onnx", ".tflite", ".engine", ".torchscript", ".mlpackage", ".pb"}
        if path.is_file() and path.suffix.lower() not in supported:
            raise RuntimeError(f"暂不支持该模型格式：{path.suffix}。可加载 PT、ONNX、TFLite、TensorRT Engine、TorchScript 等模型。")
        self.custom_tflite = None
        self.input_size = None
        if path.suffix.lower() == ".tflite":
            adapter = _CustomYolo3TFLite.try_load(path, class_names or [])
            if adapter is not None:
                self.custom_tflite = adapter
                self.model = adapter
                self.input_size = (adapter.input_width, adapter.input_height)
                self.model_path = path
                self.model_description = f"自定义 YOLOv3 TFLite · {adapter.input_width}×{adapter.input_height} · {adapter.input_dtype_name} · {len(adapter.class_names)} 类"
                return
        self.model = YOLO(str(path))
        self.model_path = path
        self.model_description = f"Ultralytics {path.suffix.upper().lstrip('.')} 模型"

    @property
    def can_train(self) -> bool:
        return self.custom_tflite is None and self.model_path is not None and self.model_path.suffix.lower() == ".pt"

    def predict(self, image_path: Path, confidence: float = 0.25) -> list[AnnotationBox]:
        if self.model is None:
            raise RuntimeError("请先加载检测模型（PT/ONNX/TFLite/Engine 等）")
        if self.custom_tflite is not None:
            return self.custom_tflite.predict(image_path, confidence)
        kwargs = {"source": str(image_path), "conf": confidence, "verbose": False}
        if self.input_size:
            kwargs["imgsz"] = self.input_size
        result = self.model.predict(**kwargs)[0]
        names = result.names
        output: list[AnnotationBox] = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            output.append(AnnotationBox(
                str(names[class_id]), round(x1), round(y1), round(x2), round(y2),
                "ai", float(box.conf[0].item()),
            ))
        return output

    @staticmethod
    def _iou(left: AnnotationBox, right: AnnotationBox) -> float:
        x1, y1 = max(left.xmin, right.xmin), max(left.ymin, right.ymin)
        x2, y2 = min(left.xmax, right.xmax), min(left.ymax, right.ymax)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        left_area = max(0, left.xmax - left.xmin) * max(0, left.ymax - left.ymin)
        right_area = max(0, right.xmax - right.xmin) * max(0, right.ymax - right.ymin)
        union = left_area + right_area - intersection
        return intersection / union if union else 0.0

    def audit_annotations(
        self, item: ImageItem, confidence: float = 0.25, iou_threshold: float = 0.5,
        missing_confidence: float | None = None, class_confidence: float | None = None,
        class_mismatch_iou: float = 0.25, report_unmatched: bool = True,
        ignore_edge_missing: bool = False,
    ) -> dict:
        """Compare saved annotations with current model predictions and return review candidates."""
        predictions = self.predict(item.path, confidence)
        missing_confidence = confidence if missing_confidence is None else missing_confidence
        class_confidence = confidence if class_confidence is None else class_confidence
        issues: list[dict] = []
        used_predictions: set[int] = set()
        for annotation_index, annotation in enumerate(item.boxes, 1):
            candidates = [(self._iou(annotation, prediction), index, prediction) for index, prediction in enumerate(predictions) if index not in used_predictions]
            # Prefer a reasonably overlapping prediction of the same class.
            # Otherwise a slightly larger wrong-class prediction can steal the
            # match and create a false wrong_class finding.
            same_class = [value for value in candidates if value[2].label == annotation.label and value[0] >= 0.20]
            pool = same_class or candidates
            best_iou, prediction_index, prediction = max(pool, default=(0.0, -1, None), key=lambda value: value[0])
            if prediction is None or best_iou < 0.15:
                if report_unmatched:
                    issues.append({
                        "type": "model_unmatched_annotation", "severity": "medium", "box_index": annotation_index,
                        "message": f"当前模型未找到与标注框 #{annotation_index}（{annotation.label}）对应的目标，请人工确认该框或检查模型漏检。",
                        "suggested_label": None, "suggested_bbox": [annotation.xmin, annotation.ymin, annotation.xmax, annotation.ymax],
                    })
                continue
            used_predictions.add(prediction_index)
            suggested_bbox = [prediction.xmin, prediction.ymin, prediction.xmax, prediction.ymax]
            if prediction.label != annotation.label and (prediction.confidence or 0) >= class_confidence and best_iou >= class_mismatch_iou:
                issues.append({
                    "type": "wrong_class", "severity": "high", "box_index": annotation_index,
                    "message": f"标注类别为 {annotation.label}，模型预测为 {prediction.label}（{prediction.confidence:.0%}，IoU {best_iou:.2f}）。",
                    "suggested_label": prediction.label, "suggested_bbox": suggested_bbox,
                    "model_confidence": prediction.confidence, "match_iou": best_iou,
                })
            elif best_iou < iou_threshold:
                issues.append({
                    "type": "box_mismatch", "severity": "medium", "box_index": annotation_index,
                    "message": f"标注框与模型框重合度较低（IoU {best_iou:.2f}），请检查框是否过松、过紧或偏移。",
                    "suggested_label": annotation.label, "suggested_bbox": suggested_bbox,
                })
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index in used_predictions:
                continue
            if (prediction.confidence or 0) < missing_confidence:
                continue
            margin_x, margin_y = max(2, item.width * 0.02), max(2, item.height * 0.02)
            touches_edge = prediction.xmin <= margin_x or prediction.ymin <= margin_y or prediction.xmax >= item.width-margin_x or prediction.ymax >= item.height-margin_y
            if ignore_edge_missing and touches_edge:
                continue
            issues.append({
                "type": "possible_missing_box", "severity": "high" if (prediction.confidence or 0) >= 0.7 else "medium",
                "box_index": None,
                "message": f"模型检测到未与现有标注匹配的 {prediction.label}（置信度 {prediction.confidence:.0%}），可能存在漏标。",
                "suggested_label": prediction.label,
                "suggested_bbox": [prediction.xmin, prediction.ymin, prediction.xmax, prediction.ymax],
                "model_confidence": prediction.confidence, "touches_edge": touches_edge,
            })
        return {
            "ok": not issues,
            "summary": f"模型预测 {len(predictions)} 个目标，与 {len(item.boxes)} 个现有标注比较，发现 {len(issues)} 个疑点。",
            "issues": issues,
        }

    def train_voc_project(
        self, project, epochs: int = 30, image_size: int = 640,
        progress: Callable[[str], None] | None = None,
    ) -> Path:
        """Convert labeled VOC images to a temporary YOLO dataset and run a small training job."""
        import random
        import shutil
        import tempfile
        import yaml
        import torch

        if self.model is None:
            raise RuntimeError("请先加载一个基础 .pt 模型")
        if not self.can_train:
            raise RuntimeError("当前加载的是推理模型，不能继续训练。请选择 YOLO11 预训练 PT 或之前训练得到的 best.pt。")
        labeled = [(item, [box for box in item.boxes if box.source == "human"]) for item in project.items if item.status != ImageStatus.REVIEW_ISSUE]
        labeled = [(item, boxes) for item, boxes in labeled if boxes]
        if len(labeled) < 2:
            raise RuntimeError("至少需要 2 张包含人工标注或已确认标注的图片；未确认的 AI 虚线框不会用于训练")
        classes = project.classes
        workspace = Path(tempfile.mkdtemp(prefix="avls_train_"))
        for split in ("train", "val"):
            (workspace / "images" / split).mkdir(parents=True)
            (workspace / "labels" / split).mkdir(parents=True)
        random.Random(42).shuffle(labeled)
        val_count = max(1, round(len(labeled) * 0.2))
        for index, (item, verified_boxes) in enumerate(labeled):
            split = "val" if index < val_count else "train"
            shutil.copy2(item.path, workspace / "images" / split / item.path.name)
            lines = []
            for box in verified_boxes:
                if box.label not in classes:
                    continue
                x = (box.xmin + box.xmax) / 2 / item.width
                y = (box.ymin + box.ymax) / 2 / item.height
                w = (box.xmax - box.xmin) / item.width
                h = (box.ymax - box.ymin) / item.height
                lines.append(f"{classes.index(box.label)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            (workspace / "labels" / split / f"{item.path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        data = {"path": str(workspace), "train": "images/train", "val": "images/val", "names": dict(enumerate(classes))}
        data_path = workspace / "data.yaml"
        data_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        if progress:
            progress(f"训练数据已准备：{len(labeled)} 张（仅人工或已确认标注）")
        device = 0 if torch.cuda.is_available() else "cpu"
        if progress:
            progress(f"训练设备：{'CUDA GPU 0 - ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
        try:
            result = self.model.train(data=str(data_path), epochs=epochs, imgsz=image_size, device=device,
                workers=2 if device == 0 else 0, project=str(project.root / ".avls" / "runs"), name="quick_train")
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
        best = Path(result.save_dir) / "weights" / "best.pt"
        self.load(best)
        return best


class _CustomYolo3TFLite:
    """Adapter for the OpenART single-head YOLOv3 int8 TFLite model."""

    def __init__(self, path, interpreter, input_detail, output_detail, anchors, class_names, nms_iou=0.45):
        self.path = path; self.interpreter = interpreter; self.input_detail = input_detail; self.output_detail = output_detail
        shape = input_detail["shape"]
        self.input_height, self.input_width = int(shape[1]), int(shape[2])
        self.input_dtype_name = input_detail["dtype"].__name__
        self.anchors = anchors; self.class_names = class_names; self.nms_iou = nms_iou

    @classmethod
    def try_load(cls, path: Path, fallback_classes: list[str]):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            return None
        interpreter = Interpreter(model_path=str(path))
        try:
            interpreter.allocate_tensors()
        except RuntimeError as exc:
            if "YOLO_Detection_PostProcess" in str(exc) or "unresolved custom op" in str(exc):
                raise RuntimeError(
                    "该 TFLite 是 OpenART 设备专用的 with_post_processing 版本，包含电脑端 LiteRT 不支持的 "
                    "YOLO_Detection_PostProcess 自定义算子。请加载同目录下不带 with_post_processing 的 "
                    "yolo3_iou_smartcar_final.tflite 进行电脑端质检。"
                ) from exc
            raise
        inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1: return None
        in_shape, out_shape = inputs[0]["shape"], outputs[0]["shape"]
        if len(in_shape) != 4 or len(out_shape) != 4 or int(out_shape[-1]) % 3 != 0: return None
        num_classes = int(out_shape[-1]) // 3 - 5
        if num_classes <= 0: return None
        config, anchors = cls._companion_config(path)
        names_text = config.get("model", "class_names", fallback="") if config else ""
        class_names = [name.strip() for name in names_text.split(",") if name.strip()]
        if len(class_names) != num_classes:
            class_names = list(fallback_classes[:num_classes])
        if len(class_names) != num_classes:
            class_names = [f"class_{index}" for index in range(num_classes)]
        if anchors is None or len(anchors) != 3:
            raise RuntimeError(
                f"识别到自定义 YOLOv3 TFLite（{in_shape.tolist()} → {out_shape.tolist()}），但未找到配套的 "
                "yolo3_anchors.txt。请把 anchor 文件和 config.cfg 放在模型同目录。"
            )
        nms_iou = config.getfloat("inference", "nms_iou_threshold", fallback=0.45) if config else 0.45
        return cls(path, interpreter, inputs[0], outputs[0], anchors, class_names, nms_iou)

    @staticmethod
    def _companion_config(path: Path):
        config = configparser.ConfigParser()
        candidates = [path.parent / "config.cfg", path.parent.parent / "config.cfg"]
        config_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if config_path: config.read(config_path, encoding="utf-8")
        anchor_candidates = [path.parent / "yolo3_anchors.txt"]
        if config.has_option("train", "cluster_anchor"):
            anchor_candidates.insert(0, Path(config.get("train", "cluster_anchor")))
        anchor_path = next((candidate for candidate in anchor_candidates if candidate.exists()), None)
        anchors = None
        if anchor_path:
            try:
                values = [float(value.strip()) for value in anchor_path.read_text(encoding="utf-8-sig").split(",") if value.strip()]
                anchors = [(values[index], values[index + 1]) for index in range(0, len(values), 2)]
            except (OSError, ValueError, IndexError): anchors = None
        return config, anchors

    @staticmethod
    def _sigmoid(value):
        value = max(-60.0, min(60.0, float(value)))
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _iou(left, right):
        x1, y1 = max(left[0], right[0]), max(left[1], right[1]); x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
        union = max(0.0, left[2]-left[0])*max(0.0, left[3]-left[1]) + max(0.0, right[2]-right[0])*max(0.0, right[3]-right[1]) - intersection
        return intersection / union if union else 0.0

    def predict(self, image_path: Path, confidence: float):
        import numpy as np
        from PIL import Image
        with Image.open(image_path) as source:
            source = source.convert("RGB"); image_width, image_height = source.size
            scale = min(self.input_width / image_width, self.input_height / image_height)
            resized_width, resized_height = max(1, round(image_width*scale)), max(1, round(image_height*scale))
            resized = source.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (self.input_width, self.input_height), (128,128,128))
            dx, dy = (self.input_width-resized_width)//2, (self.input_height-resized_height)//2
            canvas.paste(resized, (dx,dy)); float_input = np.asarray(canvas, dtype=np.float32) / 255.0
        dtype = self.input_detail["dtype"]; quant_scale, zero_point = self.input_detail.get("quantization", (0.0, 0))
        if np.issubdtype(dtype, np.integer):
            if not quant_scale: raise RuntimeError("TFLite 输入是整数类型，但模型缺少量化 scale")
            info = np.iinfo(dtype); tensor = np.clip(np.rint(float_input/quant_scale + zero_point), info.min, info.max).astype(dtype)
        else: tensor = float_input.astype(dtype)
        self.interpreter.set_tensor(self.input_detail["index"], tensor[None, ...]); self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_detail["index"])[0]
        out_scale, out_zero = self.output_detail.get("quantization", (0.0, 0))
        if np.issubdtype(raw.dtype, np.integer) and out_scale: raw = (raw.astype(np.float32)-out_zero)*out_scale
        grid_h, grid_w, channels = raw.shape; values = raw.reshape(grid_h, grid_w, 3, channels//3)
        candidates = []
        for row in range(grid_h):
            for col in range(grid_w):
                for anchor_index, (anchor_w, anchor_h) in enumerate(self.anchors):
                    entry = values[row,col,anchor_index]; objectness = self._sigmoid(entry[4])
                    class_probs = [self._sigmoid(value) for value in entry[5:]]; class_id = int(np.argmax(class_probs)); score = objectness*class_probs[class_id]
                    if score < confidence: continue
                    cx = (col+self._sigmoid(entry[0]))/grid_w*self.input_width; cy = (row+self._sigmoid(entry[1]))/grid_h*self.input_height
                    width = anchor_w*math.exp(max(-10.0,min(10.0,float(entry[2])))); height = anchor_h*math.exp(max(-10.0,min(10.0,float(entry[3]))))
                    bbox = [max(0,(cx-width/2-dx)/scale), max(0,(cy-height/2-dy)/scale), min(image_width,(cx+width/2-dx)/scale), min(image_height,(cy+height/2-dy)/scale)]
                    if bbox[2]-bbox[0] >= 2 and bbox[3]-bbox[1] >= 2: candidates.append((bbox,score,class_id))
        kept = []
        for class_id in range(len(self.class_names)):
            class_candidates = sorted((entry for entry in candidates if entry[2]==class_id), key=lambda entry: entry[1], reverse=True)
            while class_candidates and len(kept) < 100:
                best = class_candidates.pop(0); kept.append(best)
                class_candidates = [entry for entry in class_candidates if self._iou(best[0],entry[0]) < self.nms_iou]
        return [AnnotationBox(self.class_names[class_id], round(box[0]), round(box[1]), round(box[2]), round(box[3]), "ai", float(score)) for box,score,class_id in kept]
