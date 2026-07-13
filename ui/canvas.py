from __future__ import annotations

import hashlib

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget

from core.models import AnnotationBox, ImageItem


class AnnotationCanvas(QWidget):
    boxesChanged = Signal()
    selectionChanged = Signal(int)
    classCycleRequested = Signal(int)
    classNumberRequested = Signal(int)
    confirmImageRequested = Signal()
    auditIssueSelected = Signal(int)
    applyAuditIssueRequested = Signal()
    applyRecommendedAuditIssueRequested = Signal()
    deleteAuditIssueRequested = Signal()
    ignoreAuditIssueRequested = Signal()

    HANDLE = 8
    CLASS_COLORS = (
        "#42d77d", "#4ea1ff", "#ffb547", "#e66bff", "#ff637d", "#35d6d0",
        "#a5dc48", "#8b83ff", "#ff8a4c", "#42c4ff", "#e6cf48", "#f06bb5",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.item: ImageItem | None = None
        self.pixmap = QPixmap()
        self.scale = 1.0
        self.offset = QPointF()
        self.selected = -1
        self.mode = "idle"
        self.start_image = QPointF()
        self.last_widget = QPointF()
        self.original_box: AnnotationBox | None = None
        self.current_label = "object"
        self.crosshair_enabled = True
        self.mouse_widget_pos: QPointF | None = None
        self.selected_audit_issue = -1

    def set_item(self, item: ImageItem | None) -> None:
        self.item = item
        self.selected = -1
        self.selected_audit_issue = -1
        if item:
            image = QImage(str(item.path))
            self.pixmap = QPixmap.fromImage(image)
            self.fit_to_view()
        else:
            self.pixmap = QPixmap()
        self.update()

    def fit_to_view(self) -> None:
        if self.pixmap.isNull():
            return
        margin = 24
        self.scale = min((self.width() - margin * 2) / self.pixmap.width(), (self.height() - margin * 2) / self.pixmap.height())
        self.scale = max(0.05, self.scale)
        self.offset = QPointF((self.width() - self.pixmap.width() * self.scale) / 2, (self.height() - self.pixmap.height() * self.scale) / 2)

    def image_to_widget(self, point: QPointF) -> QPointF:
        return point * self.scale + self.offset

    def widget_to_image(self, point: QPointF) -> QPointF:
        return (point - self.offset) / self.scale

    def box_rect(self, box: AnnotationBox) -> QRectF:
        p1 = self.image_to_widget(QPointF(box.xmin, box.ymin))
        p2 = self.image_to_widget(QPointF(box.xmax, box.ymax))
        return QRectF(p1, p2).normalized()

    @classmethod
    def color_for_label(cls, label: str) -> QColor:
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        return QColor(cls.CLASS_COLORS[int.from_bytes(digest[:2], "big") % len(cls.CLASS_COLORS)])

    def _hit(self, pos: QPointF) -> tuple[int, str]:
        if not self.item:
            return -1, ""
        for index in range(len(self.item.boxes) - 1, -1, -1):
            rect = self.box_rect(self.item.boxes[index])
            corners = {"nw": rect.topLeft(), "ne": rect.topRight(), "sw": rect.bottomLeft(), "se": rect.bottomRight()}
            for name, point in corners.items():
                if QRectF(point.x() - self.HANDLE, point.y() - self.HANDLE, self.HANDLE * 2, self.HANDLE * 2).contains(pos):
                    return index, name
            if rect.contains(pos):
                return index, "move"
        return -1, ""

    def _audit_issue_bbox(self, issue: dict) -> list | None:
        if not self.item:
            return None
        bbox = issue.get("suggested_bbox")
        box_index = issue.get("box_index")
        if not bbox and isinstance(box_index, int) and 1 <= box_index <= len(self.item.boxes):
            box = self.item.boxes[box_index - 1]
            bbox = [box.xmin, box.ymin, box.xmax, box.ymax]
        return bbox if isinstance(bbox, list) and len(bbox) == 4 else None

    def _audit_issue_hit(self, pos: QPointF) -> int:
        if not self.item:
            return -1
        hits = []
        for index, issue in enumerate(self.item.audit_issues):
            bbox = self._audit_issue_bbox(issue)
            if not bbox:
                continue
            try:
                p1 = self.image_to_widget(QPointF(float(bbox[0]), float(bbox[1])))
                p2 = self.image_to_widget(QPointF(float(bbox[2]), float(bbox[3])))
            except (TypeError, ValueError):
                continue
            rect = QRectF(p1, p2).normalized()
            if rect.contains(pos):
                hits.append((rect.width() * rect.height(), index))
        return min(hits, default=(0, -1))[1]

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.item or event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus()
        pos = event.position()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.mode, self.last_widget = "pan", pos
            return
        issue_index = self._audit_issue_hit(pos)
        self.selected_audit_issue = issue_index
        self.auditIssueSelected.emit(issue_index)
        if issue_index >= 0:
            issue = self.item.audit_issues[issue_index]
            linked_box = issue.get("box_index")
            if isinstance(linked_box, int) and 1 <= linked_box <= len(self.item.boxes):
                self.selected = linked_box - 1
                self.selectionChanged.emit(self.selected)
                self.mode = "idle"
                self.update()
                return
            else:
                self.selected = -1
                self.selectionChanged.emit(-1)
                self.mode = "idle"
                self.update()
                return
        index, hit = self._hit(pos)
        self.selected = index
        self.selectionChanged.emit(index)
        self.start_image = self.widget_to_image(pos)
        self.last_widget = pos
        if index >= 0:
            self.mode = hit
            b = self.item.boxes[index]
            self.original_box = AnnotationBox(b.label, b.xmin, b.ymin, b.xmax, b.ymax, b.source, b.confidence)
        else:
            self.mode = "create"
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.mouse_widget_pos = event.position()
        self.update()
        if not self.item or self.mode == "idle":
            return
        pos = event.position()
        if self.mode == "pan":
            self.offset += pos - self.last_widget
            self.last_widget = pos
            self.update()
            return
        current = self.widget_to_image(pos)
        if self.mode == "create":
            self.update()
            self.last_widget = pos
            return
        if self.selected < 0 or not self.original_box:
            return
        b = self.item.boxes[self.selected]
        dx, dy = current.x() - self.start_image.x(), current.y() - self.start_image.y()
        o = self.original_box
        if self.mode == "move":
            width, height = o.xmax - o.xmin, o.ymax - o.ymin
            b.xmin = round(max(0, min(self.item.width - width, o.xmin + dx)))
            b.ymin = round(max(0, min(self.item.height - height, o.ymin + dy)))
            b.xmax, b.ymax = b.xmin + width, b.ymin + height
        else:
            if "n" in self.mode: b.ymin = round(current.y())
            if "s" in self.mode: b.ymax = round(current.y())
            if "w" in self.mode: b.xmin = round(current.x())
            if "e" in self.mode: b.xmax = round(current.x())
        self.item.dirty = True
        self.boxesChanged.emit()
        self.update()

    def leaveEvent(self, event) -> None:
        self.mouse_widget_pos = None
        self.update()
        super().leaveEvent(event)

    def set_crosshair_enabled(self, enabled: bool) -> None:
        self.crosshair_enabled = enabled
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self.item or event.button() != Qt.MouseButton.LeftButton:
            return
        if self.mode == "create":
            end = self.widget_to_image(event.position())
            box = AnnotationBox(self.current_label, round(self.start_image.x()), round(self.start_image.y()), round(end.x()), round(end.y())).normalized(self.item.width, self.item.height)
            if box.valid and box.xmax - box.xmin >= 3 and box.ymax - box.ymin >= 3:
                self.item.boxes.append(box)
                self.item.dirty = True
                self.selected = len(self.item.boxes) - 1
                self.selectionChanged.emit(self.selected)
                self.boxesChanged.emit()
        elif self.mode not in ("idle", "pan") and self.selected >= 0:
            self.item.boxes[self.selected] = self.item.boxes[self.selected].normalized(self.item.width, self.item.height)
            self.boxesChanged.emit()
        self.mode = "idle"
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.pixmap.isNull(): return
        before = self.widget_to_image(event.position())
        self.scale = max(0.05, min(20.0, self.scale * (1.15 if event.angleDelta().y() > 0 else 1 / 1.15)))
        self.offset = event.position() - before * self.scale
        self.update()

    def delete_selected(self) -> None:
        if self.item and 0 <= self.selected < len(self.item.boxes):
            self.item.boxes.pop(self.selected)
            self.selected = -1
            self.item.dirty = True
            self.selectionChanged.emit(-1)
            self.boxesChanged.emit()
            self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Delete:
            if self.item and 0 <= self.selected_audit_issue < len(self.item.audit_issues):
                self.deleteAuditIssueRequested.emit()
            else:
                self.delete_selected()
        elif event.key() == Qt.Key.Key_Q:
            self.classCycleRequested.emit(-1)
        elif event.key() == Qt.Key.Key_E:
            self.classCycleRequested.emit(1)
        elif Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_9:
            self.classNumberRequested.emit(event.key() - Qt.Key.Key_1)
        elif event.key() == Qt.Key.Key_V:
            self.confirmImageRequested.emit()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.applyAuditIssueRequested.emit()
        elif event.key() == Qt.Key.Key_W:
            self.applyRecommendedAuditIssueRequested.emit()
        elif event.key() == Qt.Key.Key_R:
            self.ignoreAuditIssueRequested.emit()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#151923"))
        if self.pixmap.isNull():
            painter.setPen(QColor("#8792a8"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "欢迎使用 QSflow\n\nCtrl+N 创建 VOC / YOLO / COCO 项目\nCtrl+O 打开已有项目\nF1 查看完整帮助",
            )
            return
        target = QRectF(self.offset, self.pixmap.size() * self.scale)
        painter.drawPixmap(target, self.pixmap, QRectF(self.pixmap.rect()))
        if self.item:
            for i, box in enumerate(self.item.boxes):
                color = self.color_for_label(box.label)
                style = Qt.PenStyle.DashLine if box.source == "ai" else Qt.PenStyle.SolidLine
                painter.setPen(QPen(color, 2 if i != self.selected else 3, style))
                rect = self.box_rect(box); painter.drawRect(rect)
                text = box.label + (f" {box.confidence:.0%}" if box.confidence is not None else "")
                painter.fillRect(QRectF(rect.left(), max(0, rect.top() - 22), max(70, len(text) * 8), 22), color)
                painter.setPen(QColor("#10141c")); painter.drawText(QPointF(rect.left() + 4, max(15, rect.top() - 6)), text)
                if i == self.selected:
                    painter.setBrush(color)
                    for p in (rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()):
                        painter.drawRect(QRectF(p.x()-4, p.y()-4, 8, 8))
            self._paint_audit_issues(painter)
        if self.mode == "create":
            painter.setPen(QPen(QColor("#4ea1ff"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(self.image_to_widget(self.start_image), self.last_widget).normalized())
        self._paint_crosshair(painter, target)

    def _paint_audit_issues(self, painter: QPainter) -> None:
        if not self.item or not self.item.audit_issues:
            return
        for issue_index, issue in enumerate(self.item.audit_issues, 1):
            bbox = issue.get("suggested_bbox")
            box_index = issue.get("box_index")
            if not bbox and isinstance(box_index, int) and 1 <= box_index <= len(self.item.boxes):
                box = self.item.boxes[box_index - 1]
                bbox = [box.xmin, box.ymin, box.xmax, box.ymax]
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                p1 = self.image_to_widget(QPointF(float(bbox[0]), float(bbox[1])))
                p2 = self.image_to_widget(QPointF(float(bbox[2]), float(bbox[3])))
            except (TypeError, ValueError):
                continue
            rect = QRectF(p1, p2).normalized()
            selected = issue_index - 1 == self.selected_audit_issue
            painter.setPen(QPen(QColor("#ffd43b") if selected else QColor("#ff3030"), 6 if selected else 4, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
            text = f"问题 {issue_index}: {issue.get('type', 'other')}"
            label = QRectF(rect.left(), max(0, rect.top() - 25), max(120, len(text) * 8), 24)
            painter.fillRect(label, QColor(220, 25, 35, 225))
            painter.setPen(QColor("white"))
            painter.drawText(label, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_crosshair(self, painter: QPainter, image_rect: QRectF) -> None:
        if not self.crosshair_enabled or self.mouse_widget_pos is None or not image_rect.contains(self.mouse_widget_pos):
            return
        pos = self.mouse_widget_pos
        painter.save()
        painter.setClipRect(image_rect)
        painter.setPen(QPen(QColor(0, 0, 0, 150), 3, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(image_rect.left(), pos.y()), QPointF(image_rect.right(), pos.y()))
        painter.drawLine(QPointF(pos.x(), image_rect.top()), QPointF(pos.x(), image_rect.bottom()))
        painter.setPen(QPen(QColor(80, 220, 255, 220), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(image_rect.left(), pos.y()), QPointF(image_rect.right(), pos.y()))
        painter.drawLine(QPointF(pos.x(), image_rect.top()), QPointF(pos.x(), image_rect.bottom()))
        image_pos = self.widget_to_image(pos)
        coordinate = f"x:{round(image_pos.x())}  y:{round(image_pos.y())}"
        label_rect = QRectF(pos.x() + 10, pos.y() + 10, 105, 23)
        if label_rect.right() > image_rect.right(): label_rect.moveLeft(pos.x() - 115)
        if label_rect.bottom() > image_rect.bottom(): label_rect.moveTop(pos.y() - 33)
        painter.fillRect(label_rect, QColor(12, 18, 27, 205))
        painter.setPen(QColor("#b8f2ff"))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, coordinate)
        painter.restore()
