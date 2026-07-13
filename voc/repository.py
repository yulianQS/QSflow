from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime
import xml.etree.ElementTree as ET
import yaml

from PIL import Image

from core.models import AnnotationBox, ImageItem, ImageStatus, Project

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _same_file_content(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()
    return digest(left) == digest(right)


def import_images(project: Project, sources: list[Path]) -> tuple[int, int, list[Path]]:
    """Recursively copy source images into the flat VOC JPEGImages directory.

    Returns (imported_count, skipped_duplicate_count, destination_paths).
    Name collisions receive a numeric suffix; byte-identical files are skipped.
    """
    discovered: list[Path] = []
    destination_root = project.images_dir.resolve()
    for source in sources:
        source = source.resolve()
        if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            discovered.append(source)
        elif source.is_dir():
            for candidate in source.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                    # Avoid importing the destination back into itself.
                    try:
                        candidate.resolve().relative_to(destination_root)
                        continue
                    except ValueError:
                        discovered.append(candidate.resolve())

    imported, skipped = 0, 0
    outputs: list[Path] = []
    project.images_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(set(discovered)):
        try:
            with Image.open(source) as image:
                image.verify()
        except (OSError, ValueError):
            skipped += 1
            continue
        target = project.images_dir / source.name
        if target.exists():
            if _same_file_content(source, target):
                skipped += 1
                continue
            index = 1
            while True:
                candidate = project.images_dir / f"{source.stem}_{index}{source.suffix.lower()}"
                if not candidate.exists():
                    target = candidate
                    break
                if _same_file_content(source, candidate):
                    target = candidate
                    skipped += 1
                    break
                index += 1
            if target.exists():
                continue
        shutil.copy2(source, target)
        imported += 1
        outputs.append(target)
    return imported, skipped, outputs


def remove_image(project: Project, item: ImageItem) -> tuple[Path | None, Path | None]:
    """Remove an image from the active VOC project into a recoverable trash folder."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trash = project.root / ".avls" / "trash" / stamp
    trash_images = trash / project.images_dir.name
    trash_annotations = trash / project.annotations_dir.name
    trash_images.mkdir(parents=True, exist_ok=True)
    trash_annotations.mkdir(parents=True, exist_ok=True)

    moved_image: Path | None = None
    moved_xml: Path | None = None
    if item.path.exists():
        moved_image = trash_images / item.path.name
        shutil.move(str(item.path), str(moved_image))
    suffix = ".xml" if project.dataset_format == "voc" else ".txt"
    annotation_path = project.annotations_dir / f"{item.path.stem}{suffix}"
    if project.dataset_format != "coco" and annotation_path.exists():
        moved_xml = trash_annotations / annotation_path.name
        shutil.move(str(annotation_path), str(moved_xml))
    project.items = [candidate for candidate in project.items if candidate.path != item.path]
    if project.dataset_format == "coco":
        if project.items: _save_coco_project(project, project.items[0])
        else: (project.annotations_dir / "instances.json").write_text(json.dumps({"images": [], "annotations": [], "categories": []}, indent=2), encoding="utf-8")
    return moved_image, moved_xml


def _write_project_metadata(project: Project) -> None:
    folder = project.root / ".avls"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "project.json").write_text(json.dumps({"format": project.dataset_format, "version": 1}, ensure_ascii=False, indent=2), encoding="utf-8")


def _item_state_path(project: Project, item: ImageItem) -> Path:
    return project.root / ".avls" / "item_state" / f"{item.path.stem}.json"


def _save_item_state(project: Project, item: ImageItem) -> None:
    path = _item_state_path(project, item); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": item.status.value, "audit_human_reviewed": item.audit_human_reviewed, "boxes": [{"source": box.source, "confidence": box.confidence} for box in item.boxes]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _restore_item_state(project: Project, item: ImageItem) -> None:
    path = _item_state_path(project, item)
    if not path.exists(): return
    try: payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError): return
    for box, state in zip(item.boxes, payload.get("boxes", [])):
        box.source = state.get("source", box.source); box.confidence = state.get("confidence", box.confidence)
    item.audit_human_reviewed = bool(payload.get("audit_human_reviewed", item.audit_human_reviewed))
    try: item.status = ImageStatus(payload.get("status", item.status.value))
    except ValueError: pass


def create_project(root: Path, dataset_format: str = "voc") -> Project:
    root.mkdir(parents=True, exist_ok=True)
    project = Project(root.resolve(), dataset_format=dataset_format)
    project.images_dir.mkdir(exist_ok=True)
    project.annotations_dir.mkdir(exist_ok=True)
    _write_project_metadata(project)
    if dataset_format == "coco":
        (project.annotations_dir / "instances.json").write_text(json.dumps({"images": [], "annotations": [], "categories": []}, indent=2), encoding="utf-8")
    return load_project(root)


def save_classes(project: Project) -> Path:
    path = project.root / "classes.txt"
    path.write_text("\n".join(project.classes), encoding="utf-8")
    if project.dataset_format == "yolo":
        (project.root / "data.yaml").write_text(yaml.safe_dump({"path": str(project.root), "train": "images", "val": "images", "names": dict(enumerate(project.classes))}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _detect_format(root: Path) -> str:
    metadata = root / ".avls" / "project.json"
    if metadata.exists():
        try:
            value = json.loads(metadata.read_text(encoding="utf-8-sig")).get("format", "voc")
            if value in {"voc", "yolo", "coco"}: return value
        except (OSError, ValueError):
            pass
    if (root / "JPEGImages").is_dir() and (root / "Annotations").is_dir(): return "voc"
    if (root / "images").is_dir() and (root / "labels").is_dir(): return "yolo"
    if (root / "images").is_dir() and (root / "annotations" / "instances.json").is_file(): return "coco"
    raise ValueError("无法识别项目格式；请选择 VOC、YOLO 或 COCO 数据集根目录")


def _load_yolo_boxes(path: Path, width: int, height: int, classes: list[str]) -> list[AnnotationBox]:
    boxes = []
    if not path.exists(): return boxes
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if len(values) < 5: continue
        try:
            class_id = int(values[0]); x, y, w, h = map(float, values[1:5])
            label = classes[class_id] if 0 <= class_id < len(classes) else f"class_{class_id}"
            boxes.append(AnnotationBox(label, round((x-w/2)*width), round((y-h/2)*height), round((x+w/2)*width), round((y+h/2)*height)))
        except (ValueError, TypeError):
            continue
    return [box.normalized(width, height) for box in boxes if box.normalized(width, height).valid]


def _read_xml(path: Path) -> tuple[list[AnnotationBox], bool]:
    boxes: list[AnnotationBox] = []
    tree = ET.parse(path)
    root = tree.getroot()
    ai_only = True
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        source = obj.findtext("source", "human")
        confidence_text = obj.findtext("confidence")
        try:
            box = AnnotationBox(
                label=obj.findtext("name", "object"),
                xmin=int(float(bnd.findtext("xmin", "0"))),
                ymin=int(float(bnd.findtext("ymin", "0"))),
                xmax=int(float(bnd.findtext("xmax", "0"))),
                ymax=int(float(bnd.findtext("ymax", "0"))),
                source=source,
                confidence=float(confidence_text) if confidence_text else None,
            )
        except ValueError:
            continue
        if box.valid:
            boxes.append(box)
            ai_only = ai_only and source == "ai"
    return boxes, bool(boxes) and ai_only


def load_project(root: Path) -> Project:
    root = root.resolve()
    dataset_format = _detect_format(root)
    project = Project(root, dataset_format=dataset_format)
    images_dir, annotations_dir = project.images_dir, project.annotations_dir
    image_index_path = root / ".avls" / "image_index.json"
    try: image_index = json.loads(image_index_path.read_text(encoding="utf-8-sig")) if image_index_path.exists() else {}
    except (OSError, ValueError): image_index = {}
    updated_image_index = {}
    classes_path = root / "classes.txt"
    if classes_path.exists():
        project.update_classes(classes_path.read_text(encoding="utf-8-sig").splitlines())
    elif dataset_format == "yolo" and (root / "data.yaml").exists():
        try:
            names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8-sig")).get("names", {})
            project.update_classes(names.values() if isinstance(names, dict) else names)
        except (OSError, ValueError, AttributeError): pass
    coco = {}
    coco_annotations = {}
    if dataset_format == "coco":
        try: coco = json.loads((annotations_dir / "instances.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError): coco = {"images": [], "annotations": [], "categories": []}
        categories = {entry["id"]: entry["name"] for entry in coco.get("categories", [])}
        project.update_classes(categories.values())
        for annotation in coco.get("annotations", []): coco_annotations.setdefault(annotation.get("image_id"), []).append(annotation)
        coco_images = {entry.get("file_name"): entry for entry in coco.get("images", [])}
    for path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        try:
            stat = path.stat()
            cached = image_index.get(path.name, {})
            cache_valid = (
                cached.get("size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns
                and all(key in cached for key in ("width", "height", "depth"))
            )
            if cache_valid:
                width, height, depth = int(cached["width"]), int(cached["height"]), int(cached["depth"])
            else:
                with Image.open(path) as image:
                    width, height = image.size
                    depth = len(image.getbands())
            updated_image_index[path.name] = {
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "width": width, "height": height, "depth": depth,
            }
        except (OSError, ValueError) as exc:
            project.warnings.append(f"已跳过无法读取的图片 {path.name}：{exc}")
            continue
        if dataset_format == "voc":
            xml_path = annotations_dir / f"{path.stem}.xml"
            try:
                boxes, ai_only = _read_xml(xml_path) if xml_path.exists() else ([], False)
            except (ET.ParseError, OSError, ValueError) as exc:
                project.warnings.append(f"标注文件 {xml_path.name} 无法解析，已按未标注图片加载：{exc}")
                boxes, ai_only = [], False
        elif dataset_format == "yolo":
            boxes, ai_only = _load_yolo_boxes(annotations_dir / f"{path.stem}.txt", width, height, project.classes), False
        else:
            image_entry = coco_images.get(path.name, {}); boxes = []
            for annotation in coco_annotations.get(image_entry.get("id"), []):
                x, y, w, h = annotation.get("bbox", [0, 0, 0, 0]); label = categories.get(annotation.get("category_id"), "object")
                box = AnnotationBox(label, round(x), round(y), round(x+w), round(y+h))
                if box.valid: boxes.append(box)
            ai_only = False
        status = ImageStatus.AI_PREDICTED if ai_only else (ImageStatus.CONFIRMED if boxes else ImageStatus.UNLABELED)
        audit_issues = []
        audit_passed = False
        audit_human_reviewed = False
        report_path = root / ".avls" / "review_issues" / "reports" / f"{path.stem}.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8-sig"))
                audit_issues = report.get("issues", []) if isinstance(report, dict) else []
                audit_passed = bool(report.get("ok", False)) and not audit_issues if isinstance(report, dict) else False
                audit_human_reviewed = bool(report.get("human_reviewed", False)) if isinstance(report, dict) else False
            except (OSError, ValueError):
                audit_issues = []
        if audit_issues or audit_human_reviewed:
            status = ImageStatus.REVIEW_ISSUE
        item = ImageItem(path, width, height, depth, boxes, status, False, audit_issues, audit_passed, audit_human_reviewed)
        _restore_item_state(project, item)
        if audit_issues: item.status = ImageStatus.REVIEW_ISSUE
        item.audit_issues = _deduplicate_audit_issues(item, item.audit_issues)
        project.items.append(item)
        project.update_classes(box.label for box in boxes)
    try:
        image_index_path.parent.mkdir(parents=True, exist_ok=True)
        image_index_path.write_text(json.dumps(updated_image_index, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        project.warnings.append(f"无法保存图片索引缓存：{exc}")
    _write_project_metadata(project)
    return project


def save_audit_report(project: Project, item: ImageItem, result: dict) -> Path:
    reports = project.root / ".avls" / "review_issues" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload["issues"] = filter_ignored_audit_issues(project, item, _deduplicate_audit_issues(item, payload.get("issues", [])))
    payload["ok"] = bool(payload.get("ok", not payload["issues"])) and not payload["issues"]
    payload["image"] = item.path.name
    payload["image_size"] = [item.width, item.height]
    payload["human_reviewed"] = bool(item.audit_human_reviewed)
    payload["boxes"] = [
        {"index": index, "label": box.label, "bbox": [box.xmin, box.ymin, box.xmax, box.ymax]}
        for index, box in enumerate(item.boxes, 1)
    ]
    path = reports / f"{item.path.stem}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    item.audit_issues = payload.get("issues", [])
    item.audit_passed = payload["ok"]
    return path


def _review_ignores_path(project: Project, item: ImageItem) -> Path:
    return project.root / ".avls" / "review_ignores" / f"{item.path.stem}.json"


def ignore_audit_issue(project: Project, item: ImageItem, issue: dict) -> None:
    path = _review_ignores_path(project, item); path.parent.mkdir(parents=True, exist_ok=True)
    try: entries = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else []
    except (OSError, ValueError): entries = []
    bbox = issue.get("suggested_bbox")
    linked = issue.get("box_index")
    if (not isinstance(bbox, list) or len(bbox) != 4) and isinstance(linked, int) and 1 <= linked <= len(item.boxes):
        box = item.boxes[linked-1]; bbox = [box.xmin, box.ymin, box.xmax, box.ymax]
    entry = {"type": issue.get("type"), "suggested_label": issue.get("suggested_label"), "bbox": bbox}
    if entry not in entries: entries.append(entry)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def filter_ignored_audit_issues(project: Project, item: ImageItem, issues: list[dict]) -> list[dict]:
    path = _review_ignores_path(project, item)
    if not path.exists(): return issues
    try: ignored = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError): return issues
    def bbox(issue):
        value = issue.get("suggested_bbox"); linked = issue.get("box_index")
        if (not isinstance(value, list) or len(value) != 4) and isinstance(linked, int) and 1 <= linked <= len(item.boxes):
            box = item.boxes[linked-1]; value = [box.xmin, box.ymin, box.xmax, box.ymax]
        return value if isinstance(value, list) and len(value) == 4 else None
    def iou(left, right):
        if not left or not right: return 0.0
        inter = max(0,min(left[2],right[2])-max(left[0],right[0]))*max(0,min(left[3],right[3])-max(left[1],right[1]))
        union = max(0,left[2]-left[0])*max(0,left[3]-left[1])+max(0,right[2]-right[0])*max(0,right[3]-right[1])-inter
        return inter/union if union else 0.0
    output = []
    for issue in issues:
        matched = any(
            issue.get("type") == entry.get("type") and
            issue.get("suggested_label") == entry.get("suggested_label") and
            iou(bbox(issue), entry.get("bbox")) >= 0.70
            for entry in ignored
        )
        if not matched: output.append(issue)
    return output


def _deduplicate_audit_issues(item: ImageItem, issues: list[dict]) -> list[dict]:
    severity_rank = {"high": 3, "medium": 2, "low": 1}
    type_rank = {"wrong_class": 6, "possible_missing_box": 5, "box_mismatch": 4, "invalid_box": 3, "model_unmatched_annotation": 2}

    def bbox(issue):
        value = issue.get("suggested_bbox")
        linked = issue.get("box_index")
        if (not isinstance(value, list) or len(value) != 4) and isinstance(linked, int) and 1 <= linked <= len(item.boxes):
            box = item.boxes[linked - 1]
            value = [box.xmin, box.ymin, box.xmax, box.ymax]
        try:
            return [float(v) for v in value] if isinstance(value, list) and len(value) == 4 else None
        except (TypeError, ValueError):
            return None

    def iou(left, right):
        if not left or not right: return 0.0
        intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))
        area_left = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        area_right = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        union = area_left + area_right - intersection
        return intersection / union if union else 0.0

    ordered = sorted(issues, key=lambda issue: (severity_rank.get(issue.get("severity"), 0), type_rank.get(issue.get("type"), 0)), reverse=True)
    kept: list[dict] = []
    for issue in ordered:
        linked = issue.get("box_index")
        issue_bbox = bbox(issue)
        duplicate = False
        for existing in kept:
            same_link = isinstance(linked, int) and linked == existing.get("box_index")
            strongly_overlapping = iou(issue_bbox, bbox(existing)) >= 0.80
            if same_link or strongly_overlapping:
                duplicate = True
                break
        if not duplicate:
            kept.append(dict(issue))
    return kept


def clear_audit_report(project: Project, item: ImageItem) -> None:
    report = project.root / ".avls" / "review_issues" / "reports" / f"{item.path.stem}.json"
    issue_image = project.root / ".avls" / "review_issues" / "images" / f"{item.path.stem}_issues.jpg"
    report.unlink(missing_ok=True)
    issue_image.unlink(missing_ok=True)
    item.audit_issues = []
    item.audit_passed = False
    item.audit_human_reviewed = False


def save_item(project: Project, item: ImageItem) -> Path:
    if project.dataset_format == "yolo":
        project.annotations_dir.mkdir(exist_ok=True)
        lines = []
        for box in item.boxes:
            box = box.normalized(item.width, item.height)
            if not box.valid or box.label not in project.classes: continue
            x = (box.xmin + box.xmax) / 2 / item.width; y = (box.ymin + box.ymax) / 2 / item.height
            w = (box.xmax - box.xmin) / item.width; h = (box.ymax - box.ymin) / item.height
            lines.append(f"{project.classes.index(box.label)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        path = project.annotations_dir / f"{item.path.stem}.txt"
        path.write_text("\n".join(lines), encoding="utf-8"); item.dirty = False; save_classes(project); _save_item_state(project, item); return path
    if project.dataset_format == "coco":
        return _save_coco_project(project, item)
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "JPEGImages"
    ET.SubElement(annotation, "filename").text = item.path.name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(item.width)
    ET.SubElement(size, "height").text = str(item.height)
    ET.SubElement(size, "depth").text = str(item.depth)
    ET.SubElement(annotation, "segmented").text = "0"
    clean_boxes = []
    for raw in item.boxes:
        box = raw.normalized(item.width, item.height)
        if not box.valid:
            continue
        clean_boxes.append(box)
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = box.label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        # Optional fields remain VOC-compatible and preserve active-learning metadata.
        ET.SubElement(obj, "source").text = box.source
        if box.confidence is not None:
            ET.SubElement(obj, "confidence").text = f"{box.confidence:.6f}"
        bnd = ET.SubElement(obj, "bndbox")
        for name, value in (("xmin", box.xmin), ("ymin", box.ymin), ("xmax", box.xmax), ("ymax", box.ymax)):
            ET.SubElement(bnd, name).text = str(value)
    item.boxes = clean_boxes
    project.annotations_dir.mkdir(exist_ok=True)
    path = project.annotations_dir / f"{item.path.stem}.xml"
    ET.indent(annotation, space="  ")
    ET.ElementTree(annotation).write(path, encoding="utf-8", xml_declaration=True)
    item.dirty = False
    _save_item_state(project, item)
    return path


def _save_coco_project(project: Project, changed_item: ImageItem) -> Path:
    project.annotations_dir.mkdir(exist_ok=True)
    categories = [{"id": index + 1, "name": name, "supercategory": "object"} for index, name in enumerate(project.classes)]
    category_ids = {entry["name"]: entry["id"] for entry in categories}
    images, annotations = [], []
    annotation_id = 1
    for image_id, item in enumerate(project.items, 1):
        images.append({"id": image_id, "file_name": item.path.name, "width": item.width, "height": item.height})
        for box in item.boxes:
            box = box.normalized(item.width, item.height)
            if not box.valid or box.label not in category_ids: continue
            width, height = box.xmax-box.xmin, box.ymax-box.ymin
            annotations.append({"id": annotation_id, "image_id": image_id, "category_id": category_ids[box.label], "bbox": [box.xmin, box.ymin, width, height], "area": width*height, "iscrowd": 0})
            annotation_id += 1
    path = project.annotations_dir / "instances.json"
    path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}, ensure_ascii=False, indent=2), encoding="utf-8")
    changed_item.dirty = False; save_classes(project); _save_item_state(project, changed_item); return path
