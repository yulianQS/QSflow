from pathlib import Path
import unittest

from core.models import AnnotationBox, ImageItem, ImageStatus, Project
from core.workflow import (
    ANNOTATION_AI, ANNOTATION_EMPTY, ANNOTATION_HUMAN, QUEUE_ALL, QUEUE_CHECK,
    QUEUE_CONFIRM, QUEUE_DONE, QUEUE_FIX, STAGE_UNLABELED, filtered_items,
    items_from_visible_paths, normalize_saved_queue, queue_counts, stage_for,
)


def item(name, status=ImageStatus.CONFIRMED, boxes=True):
    return ImageItem(Path(name), boxes=[AnnotationBox("object", 1, 1, 10, 10)] if boxes else [], status=status)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.unlabeled = item("empty.jpg", ImageStatus.UNLABELED, False)
        self.check = item("check.jpg", ImageStatus.CONFIRMED)
        self.fix = item("fix.jpg", ImageStatus.REVIEW_ISSUE); self.fix.audit_issues = [{"type": "wrong_class"}]
        self.confirm = item("confirm.jpg", ImageStatus.REVIEW_ISSUE); self.confirm.audit_human_reviewed = True
        self.done = item("done.jpg", ImageStatus.CONFIRMED); self.done.audit_human_reviewed = True
        self.ai = item("ai.jpg", ImageStatus.AI_PREDICTED)
        self.project = Project(Path("project"), [self.unlabeled, self.check, self.fix, self.confirm, self.done, self.ai])

    def test_all_review_stages(self):
        self.assertEqual(stage_for(self.unlabeled), STAGE_UNLABELED)
        self.assertEqual(stage_for(self.check), QUEUE_CHECK)
        self.assertEqual(stage_for(self.fix), QUEUE_FIX)
        self.assertEqual(stage_for(self.confirm), QUEUE_CONFIRM)
        self.assertEqual(stage_for(self.done), QUEUE_DONE)

    def test_double_filter_is_one_shared_rule(self):
        self.assertEqual(filtered_items(self.project, QUEUE_ALL, ANNOTATION_EMPTY), [self.unlabeled])
        self.assertEqual(filtered_items(self.project, QUEUE_ALL, ANNOTATION_AI), [self.ai])
        human = filtered_items(self.project, QUEUE_ALL, ANNOTATION_HUMAN)
        self.assertIn(self.check, human); self.assertIn(self.done, human); self.assertNotIn(self.ai, human)
        self.assertEqual(filtered_items(self.project, QUEUE_FIX), [self.fix])

    def test_visible_scope_never_expands_to_hidden_items(self):
        visible = items_from_visible_paths(self.project, [self.confirm.path], QUEUE_CONFIRM)
        self.assertEqual(visible, [self.confirm])
        self.assertEqual(items_from_visible_paths(self.project, [], QUEUE_CONFIRM), [])
        self.assertEqual(items_from_visible_paths(self.project, [self.done.path], QUEUE_CONFIRM), [])

    def test_counts_and_legacy_migration(self):
        counts = queue_counts(self.project)
        self.assertEqual(counts[QUEUE_FIX], 1)
        self.assertEqual(counts[QUEUE_CONFIRM], 1)
        self.assertEqual(counts[QUEUE_DONE], 1)
        self.assertEqual(normalize_saved_queue("未质检"), QUEUE_CHECK)
        self.assertEqual(normalize_saved_queue("unknown"), QUEUE_ALL)


if __name__ == "__main__":
    unittest.main()
