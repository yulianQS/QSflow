from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable


class ImageStatus(str, Enum):
    UNLABELED = "未标注"
    AI_PREDICTED = "AI预测"
    REVIEW_ISSUE = "质检有问题"
    CONFIRMED = "人工确认"


@dataclass
class AnnotationBox:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    source: str = "human"
    confidence: float | None = None

    def normalized(self, width: int, height: int) -> "AnnotationBox":
        x1, x2 = sorted((max(0, self.xmin), min(width, self.xmax)))
        y1, y2 = sorted((max(0, self.ymin), min(height, self.ymax)))
        return AnnotationBox(self.label, x1, y1, x2, y2, self.source, self.confidence)

    @property
    def valid(self) -> bool:
        return self.xmax > self.xmin and self.ymax > self.ymin


@dataclass
class ImageItem:
    path: Path
    width: int = 0
    height: int = 0
    depth: int = 3
    boxes: list[AnnotationBox] = field(default_factory=list)
    status: ImageStatus = ImageStatus.UNLABELED
    dirty: bool = False
    audit_issues: list[dict] = field(default_factory=list)
    audit_passed: bool = False
    audit_human_reviewed: bool = False

    @property
    def minimum_confidence(self) -> float:
        values = [b.confidence for b in self.boxes if b.confidence is not None]
        return min(values, default=1.0 if self.boxes else 0.0)


@dataclass
class Project:
    root: Path
    items: list[ImageItem] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    model_path: Path | None = None
    dataset_format: str = "voc"
    warnings: list[str] = field(default_factory=list)

    @property
    def images_dir(self) -> Path:
        return self.root / ("JPEGImages" if self.dataset_format == "voc" else "images")

    @property
    def annotations_dir(self) -> Path:
        return self.root / ("Annotations" if self.dataset_format == "voc" else ("labels" if self.dataset_format == "yolo" else "annotations"))

    def item_for_path(self, path: Path) -> ImageItem | None:
        return next((item for item in self.items if item.path == path), None)

    def update_classes(self, labels: Iterable[str]) -> None:
        for label in labels:
            label = label.strip()
            if label and label not in self.classes:
                self.classes.append(label)
