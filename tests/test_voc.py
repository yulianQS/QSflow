from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from core.models import AnnotationBox, ImageStatus
from voc.repository import create_project, import_images, load_project, remove_image, save_classes, save_item
from ai.vision_provider import _parse_json


class VocRepositoryTests(unittest.TestCase):
    def test_voc_round_trip(self):
        with TemporaryDirectory() as folder:
            project = create_project(Path(folder) / "dataset")
            image_path = project.images_dir / "sample.jpg"
            Image.new("RGB", (160, 120), "white").save(image_path)
            project = load_project(project.root)
            item = project.items[0]
            item.boxes = [AnnotationBox("object", 29, 56, 68, 82, "ai", 0.91)]
            item.status = ImageStatus.AI_PREDICTED
            save_item(project, item)

            loaded = load_project(project.root)
            box = loaded.items[0].boxes[0]
            self.assertEqual((box.xmin, box.ymin, box.xmax, box.ymax), (29, 56, 68, 82))
            self.assertEqual(box.label, "object")
            self.assertEqual(box.source, "ai")
            self.assertEqual(box.confidence, 0.91)
            self.assertEqual(loaded.items[0].status, ImageStatus.AI_PREDICTED)

    def test_recursive_image_import_and_name_collision(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            project = create_project(root / "dataset")
            source = root / "raw"
            (source / "bear" / "deep").mkdir(parents=True)
            (source / "tennis").mkdir(parents=True)
            Image.new("RGB", (10, 10), "red").save(source / "bear" / "deep" / "same.jpg")
            Image.new("RGB", (10, 10), "blue").save(source / "tennis" / "same.jpg")
            (source / "ignore.txt").write_text("not an image")

            imported, skipped, outputs = import_images(project, [source])
            self.assertEqual((imported, skipped), (2, 0))
            self.assertEqual({path.name for path in outputs}, {"same.jpg", "same_1.jpg"})
            self.assertEqual(len(load_project(project.root).items), 2)
            imported, skipped, _ = import_images(project, [source])
            self.assertEqual((imported, skipped), (0, 2))

    def test_remove_image_moves_image_and_xml_to_trash(self):
        with TemporaryDirectory() as folder:
            project = create_project(Path(folder) / "dataset")
            image_path = project.images_dir / "remove.jpg"
            Image.new("RGB", (20, 20), "white").save(image_path)
            project = load_project(project.root)
            project.items[0].boxes = [AnnotationBox("object", 1, 2, 10, 12)]
            save_item(project, project.items[0])
            moved_image, moved_xml = remove_image(project, project.items[0])
            self.assertFalse(image_path.exists())
            self.assertFalse((project.annotations_dir / "remove.xml").exists())
            self.assertTrue(moved_image and moved_image.exists())
            self.assertTrue(moved_xml and moved_xml.exists())
            self.assertFalse(project.items)

    def test_classes_file_persists_project_classes(self):
        with TemporaryDirectory() as folder:
            project = create_project(Path(folder) / "dataset")
            project.classes = ["bear", "tennis", "zf"]
            save_classes(project)
            self.assertEqual(load_project(project.root).classes, ["bear", "tennis", "zf"])

    def test_yolo_and_coco_round_trip(self):
        for dataset_format in ("yolo", "coco"):
            with self.subTest(dataset_format=dataset_format), TemporaryDirectory() as folder:
                project = create_project(Path(folder) / "dataset", dataset_format)
                Image.new("RGB", (100, 80), "white").save(project.images_dir / "sample.jpg")
                project = load_project(project.root)
                project.classes = ["target"]
                project.items[0].boxes = [AnnotationBox("target", 10, 12, 60, 52)]
                project.items[0].status = ImageStatus.CONFIRMED
                save_classes(project); save_item(project, project.items[0])
                loaded = load_project(project.root)
                self.assertEqual(loaded.dataset_format, dataset_format)
                self.assertEqual(loaded.classes, ["target"])
                box = loaded.items[0].boxes[0]
                self.assertEqual((box.label, box.xmin, box.ymin, box.xmax, box.ymax), ("target", 10, 12, 60, 52))

    def test_corrupt_image_and_xml_do_not_block_project(self):
        with TemporaryDirectory() as folder:
            project = create_project(Path(folder) / "dataset")
            Image.new("RGB", (20, 20), "white").save(project.images_dir / "good.jpg")
            (project.images_dir / "broken.jpg").write_bytes(b"not an image")
            (project.annotations_dir / "good.xml").write_text("<annotation><broken>", encoding="utf-8")
            loaded = load_project(project.root)
            self.assertEqual([entry.path.name for entry in loaded.items], ["good.jpg"])
            self.assertEqual(len(loaded.warnings), 2)
            self.assertFalse(loaded.items[0].boxes)

    def test_image_dimension_cache_is_used_on_reopen(self):
        with TemporaryDirectory() as folder:
            project = create_project(Path(folder) / "dataset")
            Image.new("RGB", (64, 48), "white").save(project.images_dir / "cached.jpg")
            first = load_project(project.root)
            self.assertEqual((first.items[0].width, first.items[0].height), (64, 48))
            self.assertTrue((project.root / ".avls" / "image_index.json").exists())
            with patch("voc.repository.Image.open", side_effect=AssertionError("cache miss")):
                second = load_project(project.root)
            self.assertEqual((second.items[0].width, second.items[0].height), (64, 48))

    def test_vision_audit_json_parser_accepts_fenced_json(self):
        result = _parse_json('```json\n{"ok": false, "summary": "wrong", "issues": [{"type": "wrong_class"}]}\n```')
        self.assertFalse(result["ok"])
        self.assertEqual(result["issues"][0]["type"], "wrong_class")


if __name__ == "__main__":
    unittest.main()
