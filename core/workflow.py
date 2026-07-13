"""Single source of truth for QSflow review queues and UI filtering."""

from collections.abc import Iterable

from core.models import ImageItem, ImageStatus, Project


QUEUE_ALL = "全部工作队列"
QUEUE_CHECK = "① 待检查（模型 / 人工任选）"
QUEUE_FIX = "② AI发现疑点，待人工处理"
QUEUE_CONFIRM = "③ 人工已处理，待批量确认"
QUEUE_DONE = "✓ 检查流程已完成"
STAGE_UNLABELED = "无标注框"

ANNOTATION_ALL = "全部标注状态"
ANNOTATION_EMPTY = "无标注框"
ANNOTATION_AI = "AI框待确认"
ANNOTATION_HUMAN = "人工标注已确认"

QUEUE_FILTERS = (QUEUE_ALL, QUEUE_CHECK, QUEUE_FIX, QUEUE_CONFIRM, QUEUE_DONE)
ANNOTATION_FILTERS = (ANNOTATION_ALL, ANNOTATION_EMPTY, ANNOTATION_AI, ANNOTATION_HUMAN)


def stage_for(item: ImageItem) -> str:
    if not item.boxes:
        return STAGE_UNLABELED
    if item.audit_issues:
        return QUEUE_FIX
    if item.audit_human_reviewed and item.status == ImageStatus.REVIEW_ISSUE:
        return QUEUE_CONFIRM
    if item.audit_passed or (item.audit_human_reviewed and item.status == ImageStatus.CONFIRMED):
        return QUEUE_DONE
    return QUEUE_CHECK


def matches_annotation_filter(item: ImageItem, selected: str) -> bool:
    if selected == ANNOTATION_EMPTY:
        return not item.boxes
    if selected == ANNOTATION_AI:
        return item.status == ImageStatus.AI_PREDICTED
    if selected == ANNOTATION_HUMAN:
        return item.status == ImageStatus.CONFIRMED
    return True


def filtered_items(project: Project, queue: str = QUEUE_ALL, annotation: str = ANNOTATION_ALL) -> list[ImageItem]:
    return [
        item for item in project.items
        if (queue == QUEUE_ALL or stage_for(item) == queue) and matches_annotation_filter(item, annotation)
    ]


def items_from_visible_paths(project: Project, paths: Iterable, required_stage: str) -> list[ImageItem]:
    """Resolve a UI snapshot back to backend items without expanding its scope."""
    result = []
    seen = set()
    by_path = {item.path: item for item in project.items}
    for path in paths:
        item = by_path.get(path)
        if item and item.path not in seen and stage_for(item) == required_stage:
            result.append(item)
            seen.add(item.path)
    return result


def queue_counts(project: Project) -> dict[str, int]:
    counts = {QUEUE_CHECK: 0, QUEUE_FIX: 0, QUEUE_CONFIRM: 0, QUEUE_DONE: 0, STAGE_UNLABELED: 0}
    for item in project.items:
        stage = stage_for(item)
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def normalize_saved_queue(value: str) -> str:
    migration = {
        "全部": QUEUE_ALL,
        "未质检": QUEUE_CHECK,
        "① 待模型质检": QUEUE_CHECK,
        "已质检": QUEUE_DONE,
        "✓ 质检流程已完成": QUEUE_DONE,
        "待人工修正": QUEUE_FIX,
        "质检有问题": QUEUE_FIX,
        "已复核待确认": QUEUE_CONFIRM,
        "人工确认": QUEUE_ALL,
        "未标注": QUEUE_ALL,
        "AI预测": QUEUE_ALL,
    }
    return migration.get(value, value if value in QUEUE_FILTERS else QUEUE_ALL)
