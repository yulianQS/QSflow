from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication, QMessageBox

QStandardPaths.setTestModeEnabled(True)

from core.models import AnnotationBox, ImageStatus
from core.workflow import QUEUE_CONFIRM, QUEUE_DONE
from ui.main_window import MainWindow
from voc.repository import create_project, load_project, save_item


APP = QApplication.instance() or QApplication([])
APP.setApplicationName("QSflow-Tests")
APP.setOrganizationName("K&C")


class UiWorkflowTests(unittest.TestCase):
    def test_empty_left_scope_clears_canvas_and_never_confirms_hidden_items(self):
        with TemporaryDirectory() as folder:
            project = create_project(Path(folder) / "dataset")
            Image.new("RGB", (30, 20), "white").save(project.images_dir / "done.jpg")
            project = load_project(project.root)
            item = project.items[0]
            item.boxes = [AnnotationBox("object", 1, 1, 10, 10)]
            item.status = ImageStatus.CONFIRMED
            item.audit_human_reviewed = True
            save_item(project, item)

            window = MainWindow(); window._set_project(load_project(project.root))
            window.filter_combo.setCurrentText(QUEUE_CONFIRM); APP.processEvents()
            self.assertEqual(window.image_list.count(), 0)
            self.assertIsNone(window.current)
            self.assertTrue(window.canvas.pixmap.isNull())

            information = QMessageBox.information
            question = QMessageBox.question
            calls = []
            try:
                QMessageBox.information = lambda *args, **kwargs: calls.append(args[1])
                QMessageBox.question = lambda *args, **kwargs: self.fail("Empty left scope must not ask to confirm hidden images")
                window.confirm_human_reviewed_images()
            finally:
                QMessageBox.information = information
                QMessageBox.question = question
            self.assertTrue(calls)
            self.assertEqual(window._queue_stage(item), QUEUE_DONE)
            window.deleteLater()


if __name__ == "__main__":
    unittest.main()
