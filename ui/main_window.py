from __future__ import annotations

from pathlib import Path
import json
import logging

from PySide6.QtCore import Qt, QThread, Signal, QSettings, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QTabWidget, QToolBar, QVBoxLayout, QWidget)

from ai.vision_provider import VisionAuditService, VisionProviderConfig, export_issue_image
from ai.yolo_service import YoloService
from core.app_paths import logs_dir, settings_path
from core.models import AnnotationBox, ImageStatus
from core.workflow import (
    ANNOTATION_FILTERS, QUEUE_ALL, QUEUE_CHECK as QUEUE_AUDIT, QUEUE_CONFIRM,
    QUEUE_DONE, QUEUE_FILTERS, QUEUE_FIX, STAGE_UNLABELED, filtered_items,
    items_from_visible_paths, normalize_saved_queue, queue_counts, stage_for,
)
from core.version import APP_DISPLAY_NAME
from ui.canvas import AnnotationCanvas
from ui.help_dialog import HelpDialog
from voc.repository import clear_audit_report, create_project, filter_ignored_audit_issues, ignore_audit_issue, import_images, load_project, remove_image, save_audit_report, save_classes, save_item

class ImageListWidget(QListWidget):
    deleteRequested = Signal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.deleteRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class TaskThread(QThread):
    done = Signal(object)
    failed = Signal(str)
    def __init__(self, fn): super().__init__(); self.fn = fn
    def run(self):
        try: self.done.emit(self.fn())
        except Exception as exc:
            logging.exception("Background task failed")
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.resize(1440, 900)
        self.project = None
        self.current = None
        self.yolo = YoloService()
        self.vision_audit = VisionAuditService()
        self.worker = None
        self._shown_project_warning_roots = set()
        self.settings = QSettings(str(settings_path()), QSettings.Format.IniFormat)
        self._restore_image_name = ""
        self._build_ui(); self._build_actions(); self._apply_style(); self._restore_window_state()

    def _build_ui(self):
        self.image_list = ImageListWidget(); self.image_list.currentRowChanged.connect(self._select_image); self.image_list.deleteRequested.connect(self.remove_current_image)
        self.filter_combo = QComboBox(); self.filter_combo.addItems(QUEUE_FILTERS)
        self.filter_combo.currentTextChanged.connect(self._queue_changed)
        self.annotation_filter_combo = QComboBox(); self.annotation_filter_combo.addItems(ANNOTATION_FILTERS)
        self.annotation_filter_combo.currentTextChanged.connect(self.refresh_list)
        review_legend = QLabel("工作队列与右侧①②③按钮完全对应；\n标注状态只用于进一步缩小图片范围。")
        review_legend.setWordWrap(True); review_legend.setStyleSheet("color:#9aa7bd;font-size:12px;padding:3px")
        self.queue_action_button = QPushButton("请选择①②③工作队列")
        self.queue_action_button.clicked.connect(self.run_current_queue_action)
        self.queue_action_button.setStyleSheet("QPushButton{background:#40536d;font-weight:bold} QPushButton:hover{background:#4d6380}")
        self.manual_pass_current_button = QPushButton("人工查看通过当前图片（V）")
        self.manual_pass_current_button.clicked.connect(self.confirm_current_annotations)
        self.manual_pass_queue_button = QPushButton("批量人工通过左侧图片…")
        self.manual_pass_queue_button.clicked.connect(self.confirm_left_queue_manually)
        self.queue_help_label = QLabel("先选工作队列；软件会自动切换到对应功能区。")
        self.queue_help_label.setWordWrap(True)
        self.queue_help_label.setStyleSheet("color:#b8c5d9;font-size:12px;padding:3px")
        remove_button = QPushButton("从项目移除图片"); remove_button.clicked.connect(self.remove_current_image)
        left = QWidget(); ll = QVBoxLayout(left); ll.addWidget(QLabel("第一步：选择当前要做的工作")); ll.addWidget(self.filter_combo); ll.addWidget(self.queue_action_button); ll.addWidget(self.manual_pass_current_button); ll.addWidget(self.manual_pass_queue_button); ll.addWidget(self.queue_help_label); ll.addWidget(QLabel("第二步（可选）：按标注状态缩小范围")); ll.addWidget(self.annotation_filter_combo); ll.addWidget(review_legend); ll.addWidget(self.image_list); ll.addWidget(remove_button)

        self.canvas = AnnotationCanvas(); self.canvas.boxesChanged.connect(self._boxes_changed); self.canvas.selectionChanged.connect(self._selection_changed)
        self.canvas.classCycleRequested.connect(self.cycle_selected_class); self.canvas.classNumberRequested.connect(self.set_selected_class_by_index)
        self.canvas.confirmImageRequested.connect(self.confirm_current_annotations)
        self.canvas.auditIssueSelected.connect(self._audit_issue_selected)
        self.canvas.applyAuditIssueRequested.connect(self.apply_class_to_selected_issue)
        self.canvas.applyRecommendedAuditIssueRequested.connect(self.apply_recommended_to_selected_issue)
        self.canvas.deleteAuditIssueRequested.connect(self.delete_selected_issue)
        self.canvas.ignoreAuditIssueRequested.connect(self.ignore_selected_issue)

        self.class_combo = QComboBox(); self.class_combo.setEditable(True); self.class_combo.currentTextChanged.connect(self._class_changed)
        self.coords = QLabel("未选择标注框")
        self.class_shortcuts = QLabel("快捷键：Q/E 上一类/下一类；1–9 直接选择")
        self.class_shortcuts.setWordWrap(True)
        add_class = QPushButton("添加类别"); add_class.clicked.connect(self.add_class)
        delete_class = QPushButton("删除当前类别"); delete_class.clicked.connect(self.delete_current_class)
        confirm_image = QPushButton("V  确认当前图片标注"); confirm_image.clicked.connect(self.confirm_current_annotations)
        self.issue_info = QLabel("未选择问题框"); self.issue_info.setWordWrap(True)
        apply_issue_class = QPushButton("将所选问题框应用为当前类别"); apply_issue_class.clicked.connect(self.apply_class_to_selected_issue)
        ignore_issue = QPushButton("R  标注正确，忽略此误报"); ignore_issue.clicked.connect(self.ignore_selected_issue)
        ignore_issue.setStyleSheet("QPushButton{background:#586174} QPushButton:hover{background:#687388}")
        delete_box = QPushButton("删除当前框"); delete_box.clicked.connect(self.canvas.delete_selected)
        previous_class = QPushButton("Q  上一个类别"); previous_class.clicked.connect(lambda: self.cycle_selected_class(-1))
        next_class = QPushButton("E  下一个类别"); next_class.clicked.connect(lambda: self.cycle_selected_class(1))
        class_buttons = QWidget(); class_buttons_layout = QHBoxLayout(class_buttons); class_buttons_layout.setContentsMargins(0, 0, 0, 0); class_buttons_layout.addWidget(previous_class); class_buttons_layout.addWidget(next_class)
        shortcut_help = QLabel(
            "<b>快捷键速查</b><br><br>"
            "鼠标拖动空白处　创建标注框<br>"
            "拖动框内部　　　移动标注框<br>"
            "拖动框四角　　　缩放标注框<br>"
            "Delete　　　　　删除选中框<br>"
            "Q / E　　　　　 上一类 / 下一类<br>"
            "1–9　　　　　　直接选择类别<br>"
            "A / D　　　　　 上一张 / 下一张<br>"
            "Ctrl + S　　　　保存标注<br>"
            "鼠标滚轮　　　　缩放图片<br>"
            "Ctrl + 左键拖动　平移图片<br>"
            "X　　　　　　　 开关十字虚线辅助<br><br>"
            "V　　　　　　　 确认整张图片标注<br><br>"
            "问题框：Q/E 上一类/下一类并确认<br>"
            "问题框：1–9 选类，Enter 当前类确认<br>"
            "问题框：W 推荐类别，Delete 删除/驳回<br>"
            "问题框：R 标注正确，记住并忽略误报<br><br>"
            "<span style='color:#8792a8'>提示：先点击画布或标注框，画布快捷键才会生效。</span>"
        )
        shortcut_help.setWordWrap(True)
        shortcut_help.setAlignment(Qt.AlignmentFlag.AlignTop)
        shortcut_help.setStyleSheet("QLabel{background:#171c26;border:1px solid #354052;border-radius:6px;padding:12px;line-height:1.5}")
        annotation = QWidget(); form = QFormLayout(annotation); form.addRow("类别", self.class_combo); form.addRow("快速切换", class_buttons); form.addRow("", self.class_shortcuts); form.addRow("坐标", self.coords); form.addRow("问题框", self.issue_info); form.addRow(apply_issue_class); form.addRow(ignore_issue); form.addRow(confirm_image); form.addRow(add_class); form.addRow(delete_class); form.addRow(delete_box); form.addRow(shortcut_help)

        self.confidence = QDoubleSpinBox(); self.confidence.setRange(0.01, 1); self.confidence.setSingleStep(.05); self.confidence.setValue(.25)
        self.model_label = QLabel("未加载模型")
        self.training_device_label = QLabel(self._training_device_text())
        self.training_base = QComboBox(); self.training_base.addItem("YOLO11 Nano（推荐，自动下载）", "yolo11n.pt"); self.training_base.addItem("YOLO11 Small（更准但更慢）", "yolo11s.pt"); self.training_base.addItem("YOLOv8 Nano（兼容选项）", "yolov8n.pt")
        load_model = QPushButton("加载检测模型（PT / ONNX / TFLite…）"); load_model.clicked.connect(self.load_model)
        predict_one = QPushButton("自动标注当前图片"); predict_one.clicked.connect(self.predict_current)
        predict_all = QPushButton("批量标注未标注图片"); predict_all.clicked.connect(self.predict_batch)
        train = QPushButton("用当前标注快速训练"); train.clicked.connect(self.quick_train)
        base_hint = QLabel("未手动加载模型时，快速训练会自动下载所选预训练权重。")
        base_hint.setWordWrap(True)
        ai = QWidget(); af = QFormLayout(ai); af.addRow("当前模型", self.model_label); af.addRow("训练设备", self.training_device_label); af.addRow("置信度", self.confidence); af.addRow(load_model); af.addRow("训练基础模型", self.training_base); af.addRow("", base_hint); af.addRow(predict_one); af.addRow(predict_all); af.addRow(train)

        self.provider_name = QLineEdit()
        self.provider_base_url = QLineEdit()
        self.provider_model = QLineEdit()
        self.provider_protocol = QComboBox(); self.provider_protocol.addItem("OpenAI Responses API", "responses"); self.provider_protocol.addItem("OpenAI-compatible Chat Completions", "chat_completions")
        self.provider_key_env = QLineEdit()
        self.provider_api_key = QLineEdit(); self.provider_api_key.setEchoMode(QLineEdit.EchoMode.Password); self.provider_api_key.setPlaceholderText("仅本次运行使用，不保存")
        save_provider = QPushButton("保存模型 API 配置"); save_provider.clicked.connect(self.save_provider_config)
        self.audit_profile = QComboBox(); self.audit_profile.addItem("保守（少误报，推荐）", "conservative"); self.audit_profile.addItem("平衡", "balanced"); self.audit_profile.addItem("敏感（尽量不漏疑点）", "sensitive")
        self.local_audit_scope = QComboBox(); self.local_audit_scope.addItem("仅审查未检查图片（推荐）", "unchecked"); self.local_audit_scope.addItem("重新审查全部已标注图片", "all")
        audit_trained_model = QPushButton("①A 使用模型审查左侧 [待检查]"); audit_trained_model.clicked.connect(self.audit_unchecked_images)
        manual_pass_one = QPushButton("①B 人工查看通过当前图片（V）"); manual_pass_one.clicked.connect(self.confirm_current_annotations)
        manual_pass_batch = QPushButton("①C 批量人工通过左侧 [待检查]…"); manual_pass_batch.clicked.connect(self.confirm_left_queue_manually)
        show_issues = QPushButton("② 打开左侧 [待人工处理]"); show_issues.clicked.connect(lambda: self.set_image_filter(QUEUE_FIX))
        confirm_reviewed = QPushButton("③ 一键通过左侧 [待批量确认]"); confirm_reviewed.clicked.connect(self.confirm_human_reviewed_images)
        confirm_safe_ai = QPushButton("自动审查并通过 [AI预测] 无疑点图片"); confirm_safe_ai.clicked.connect(self.confirm_audited_safe_ai)
        rescreen_issues = QPushButton("一键保守重筛现有问题队列"); rescreen_issues.clicked.connect(self.rescreen_existing_issues)
        audit_trained_model.setToolTip("只审查尚无质检记录的已标注图片；不会重复审查旧图片")
        show_issues.setToolTip("把左侧切换到仍有问题框、需要人工处理的图片")
        confirm_reviewed.setToolTip("只确认已经处理完全部问题框并带有人工复核标记的图片")
        show_issues.setStyleSheet("QPushButton{background:#b86b20} QPushButton:hover{background:#d7832c}")
        confirm_reviewed.setStyleSheet("QPushButton{background:#238b57} QPushButton:hover{background:#2ca96b}")
        audit_one = QPushButton("模型审查当前图片"); audit_one.clicked.connect(self.audit_current)
        audit_batch_model = QPushButton("模型批量审查图片…"); audit_batch_model.clicked.connect(self.choose_model_audit_scope)
        audit_confirm_codex = QPushButton("Codex 审查当前并自动确认"); audit_confirm_codex.clicked.connect(self.audit_and_confirm_with_codex)
        audit_batch_codex = QPushButton("Codex 批量审查未确认图片"); audit_batch_codex.clicked.connect(self.audit_unconfirmed_with_codex)
        self.gpt_output = QPlainTextEdit(); self.gpt_output.setReadOnly(True); self.gpt_output.setPlaceholderText("质检结果将在这里显示：漏标、错类、框偏移、重复框等。")
        self.audit_issue_list = QListWidget(); self.audit_issue_list.setMaximumHeight(150); self.audit_issue_list.itemDoubleClicked.connect(self._jump_to_audit_issue)
        gpt = QWidget(); gl = QVBoxLayout(gpt)
        provider_form = QFormLayout(); provider_form.addRow("服务名称", self.provider_name); provider_form.addRow("Base URL", self.provider_base_url); provider_form.addRow("模型名称", self.provider_model); provider_form.addRow("接口协议", self.provider_protocol); provider_form.addRow("Key 环境变量", self.provider_key_env); provider_form.addRow("API Key", self.provider_api_key)
        workflow_title = QLabel("<b>检查工作流（模型或人工均可）</b>")
        self.review_counts_label = QLabel(); self.review_counts_label.setWordWrap(True)
        self.review_counts_label.setStyleSheet("QLabel{background:#171c26;border:1px solid #354052;border-radius:6px;padding:9px;color:#b9c7dc}")
        workflow_help = QLabel("①待检查可任选：A 模型审查，或 B/C 人工直接通过；只有模型发现疑点时才进入②，处理后进入③。")
        workflow_help.setWordWrap(True); workflow_help.setStyleSheet("color:#93a4bd")
        gl.addWidget(workflow_title); gl.addWidget(self.review_counts_label); gl.addWidget(workflow_help)
        gl.addWidget(QLabel("质检敏感度")); gl.addWidget(self.audit_profile)
        gl.addWidget(audit_trained_model); gl.addWidget(manual_pass_one); gl.addWidget(manual_pass_batch); gl.addWidget(show_issues); gl.addWidget(confirm_reviewed); gl.addWidget(confirm_safe_ai)
        gl.addWidget(rescreen_issues)
        gl.addWidget(QLabel("重新检查范围")); gl.addWidget(self.local_audit_scope)
        recheck = QPushButton("按上方范围重新运行本地模型审查"); recheck.clicked.connect(self.audit_all_with_trained_model); gl.addWidget(recheck)

        api_toggle = QPushButton("高级：API / 自定义视觉模型 ▸"); api_toggle.setCheckable(True)
        api_content = QWidget(); api_layout = QVBoxLayout(api_content); api_layout.setContentsMargins(0, 0, 0, 0); api_layout.addLayout(provider_form); api_layout.addWidget(save_provider); api_layout.addWidget(audit_one); api_layout.addWidget(audit_batch_model); api_content.setVisible(False)
        api_toggle.toggled.connect(lambda checked: (api_content.setVisible(checked), api_toggle.setText("高级：API / 自定义视觉模型 ▾" if checked else "高级：API / 自定义视觉模型 ▸")))
        gl.addWidget(api_toggle); gl.addWidget(api_content)

        codex_toggle = QPushButton("高级：本机 Codex ▸"); codex_toggle.setCheckable(True)
        codex_content = QWidget(); codex_layout = QVBoxLayout(codex_content); codex_layout.setContentsMargins(0, 0, 0, 0); codex_layout.addWidget(audit_confirm_codex); codex_layout.addWidget(audit_batch_codex); codex_content.setVisible(False)
        codex_toggle.toggled.connect(lambda checked: (codex_content.setVisible(checked), codex_toggle.setText("高级：本机 Codex ▾" if checked else "高级：本机 Codex ▸")))
        gl.addWidget(codex_toggle); gl.addWidget(codex_content)
        gl.addWidget(QLabel("问题图片（双击定位）")); gl.addWidget(self.audit_issue_list); gl.addWidget(QLabel("标注质检报告")); gl.addWidget(self.gpt_output)
        self._load_provider_config()

        gpt_scroll = QScrollArea(); gpt_scroll.setWidgetResizable(True); gpt_scroll.setFrameShape(QScrollArea.Shape.NoFrame); gpt_scroll.setWidget(gpt)
        self.tabs = QTabWidget(); self.tabs.addTab(annotation, "标注"); self.tabs.addTab(ai, "AI助手"); self.tabs.addTab(gpt_scroll, "检查与质检")
        self.splitter = QSplitter(); self.splitter.addWidget(left); self.splitter.addWidget(self.canvas); self.splitter.addWidget(self.tabs); self.splitter.setSizes([260, 900, 300])
        self.setCentralWidget(self.splitter); self.statusBar().showMessage("就绪")

    def _build_actions(self):
        file_menu = self.menuBar().addMenu("文件")
        for text, slot, shortcut in (("创建数据集项目", self.create_project_dialog, "Ctrl+N"), ("打开数据集项目", self.open_project_dialog, "Ctrl+O"), ("保存", self.save_current, "Ctrl+S")):
            action = QAction(text, self); action.triggered.connect(slot); action.setShortcut(QKeySequence(shortcut)); file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction("导入图片文件…", self.import_image_files)
        file_menu.addAction("导入图片文件夹（递归）…", self.import_image_folder)
        project_menu = self.menuBar().addMenu("项目"); project_menu.addAction("添加类别", self.add_class); project_menu.addAction("删除当前类别", self.delete_current_class)
        project_menu.addAction("移除当前图片", self.remove_current_image)
        ai_menu = self.menuBar().addMenu("AI"); ai_menu.addAction("加载检测模型…", self.load_model); ai_menu.addAction("自动标注当前图片", self.predict_current)
        tools_menu = self.menuBar().addMenu("工具")
        tools_menu.addAction("适应窗口", self.canvas.fit_to_view)
        self.crosshair_action = QAction("十字虚线辅助", self); self.crosshair_action.setCheckable(True); self.crosshair_action.setChecked(True); self.crosshair_action.setShortcut(QKeySequence("X")); self.crosshair_action.toggled.connect(self.canvas.set_crosshair_enabled); tools_menu.addAction(self.crosshair_action)
        help_menu = self.menuBar().addMenu("帮助")
        help_action = QAction("使用手册", self); help_action.setShortcut(QKeySequence("F1")); help_action.triggered.connect(lambda: self.show_help(0)); help_menu.addAction(help_action)
        help_menu.addAction("标注与 AI 使用方法", lambda: self.show_help(1))
        help_menu.addAction("检查工作流说明", lambda: self.show_help(2))
        help_menu.addAction("快捷键速查", lambda: self.show_help(3))
        help_menu.addAction("常见问题", lambda: self.show_help(4))
        help_menu.addAction("打开诊断日志目录", self.open_logs_directory)
        help_menu.addSeparator(); help_menu.addAction("关于 QSflow", lambda: self.show_help(5))
        toolbar = QToolBar(); self.addToolBar(toolbar)
        for text, slot in (("打开项目", self.open_project_dialog), ("导入图片", self.import_image_folder), ("保存", self.save_current), ("上一张", lambda: self.navigate(-1)), ("下一张", lambda: self.navigate(1)), ("适应窗口", lambda: (self.canvas.fit_to_view(), self.canvas.update()))): toolbar.addAction(text, slot)
        prev = QAction(self); prev.setShortcut(QKeySequence("A")); prev.triggered.connect(lambda: self.navigate(-1)); self.addAction(prev)
        nxt = QAction(self); nxt.setShortcut(QKeySequence("D")); nxt.triggered.connect(lambda: self.navigate(1)); self.addAction(nxt)

    def show_help(self, initial_tab=0):
        HelpDialog(self, initial_tab).exec()

    def open_logs_directory(self):
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs_dir()))):
            self.error(f"无法打开日志目录：{logs_dir()}")

    def _apply_style(self):
        self.setStyleSheet("QMainWindow,QWidget{background:#202633;color:#e7ebf2;font-size:13px} QMenuBar,QMenu,QToolBar{background:#171c26} QListWidget,QLineEdit,QComboBox,QDoubleSpinBox,QPlainTextEdit{background:#151923;border:1px solid #354052;border-radius:4px;padding:5px} QPushButton{background:#3178c6;border:0;border-radius:5px;padding:8px} QPushButton:hover{background:#4090e0} QTabWidget::pane{border:1px solid #354052} QTabBar::tab{padding:8px;background:#171c26} QTabBar::tab:selected{background:#3178c6}")

    def create_project_dialog(self):
        display, ok = QInputDialog.getItem(self, "创建项目", "数据集格式", ["Pascal VOC", "YOLO Detection", "COCO Detection"], 0, False)
        if not ok: return
        dataset_format = {"Pascal VOC": "voc", "YOLO Detection": "yolo", "COCO Detection": "coco"}[display]
        path = QFileDialog.getExistingDirectory(self, f"选择新的 {display} 项目目录")
        if path:
            try: self._set_project(create_project(Path(path), dataset_format))
            except Exception as exc: self.error(str(exc))

    def open_project_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "选择 VOC、YOLO 或 COCO 数据集根目录")
        if path: self.open_project(Path(path))

    def open_project(self, path: Path):
        try: self._set_project(load_project(path))
        except Exception as exc: self.error(str(exc))

    def import_image_files(self):
        if not self._require_project(): return
        paths, _ = QFileDialog.getOpenFileNames(self, "选择要导入的图片", "", "图片 (*.jpg *.jpeg *.png *.bmp)")
        if paths: self._import_sources([Path(path) for path in paths])

    def import_image_folder(self):
        if not self._require_project(): return
        path = QFileDialog.getExistingDirectory(self, "选择图片素材文件夹（自动扫描所有子文件夹）")
        if path: self._import_sources([Path(path)])

    def _require_project(self):
        if self.project: return True
        QMessageBox.information(self, "请先打开项目", "请先创建或打开一个 VOC、YOLO 或 COCO 项目，再导入图片素材。")
        return False

    def _import_sources(self, sources):
        try:
            imported, skipped, _ = import_images(self.project, sources)
            root = self.project.root
            self._set_project(load_project(root))
            QMessageBox.information(self, "导入完成", f"成功导入：{imported} 张\n重复或无效图片跳过：{skipped} 张\n\n图片已复制到 {self.project.images_dir.name}。\n旧图片的质检记录已保留，只需审查新导入图片。")
        except Exception as exc:
            self.error(f"导入失败：{exc}")

    def remove_current_image(self):
        if not self.project or not self.current:
            return
        name = self.current.path.name
        answer = QMessageBox.question(
            self,
            "移除图片",
            f"确定从项目中移除“{name}”吗？\n\n图片和对应 XML 会移动到 .avls/trash，可以手动恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        old_row = self.image_list.currentRow()
        try:
            remove_image(self.project, self.current)
            self.current = None
            self.canvas.set_item(None)
            self.refresh_list()
            if self.image_list.count():
                self.image_list.setCurrentRow(min(old_row, self.image_list.count() - 1))
            self.statusBar().showMessage(f"已从项目移除：{name}", 5000)
        except Exception as exc:
            self.error(f"移除失败：{exc}")

    def _set_project(self, project):
        if self.project and self.current and self.current.dirty:
            self.save_current()
        if self.project:
            self._save_project_ui_state()
        self.current = None
        self.canvas.set_item(None)
        self.yolo = YoloService()
        self.model_label.setText("未加载模型")
        self.project = project
        self.settings.setValue("session/project", str(project.root))
        self.settings.sync()
        project_state = self._load_project_ui_state(project)
        remembered_filter = self._normalize_saved_queue(project_state.get("filter", self.settings.value("session/filter", QUEUE_ALL, str)))
        if self.filter_combo.findText(remembered_filter) >= 0: self.filter_combo.setCurrentText(remembered_filter)
        annotation_filter = project_state.get("annotation_filter", self.settings.value("session/annotation_filter", "全部标注状态", str))
        if self.annotation_filter_combo.findText(annotation_filter) >= 0: self.annotation_filter_combo.setCurrentText(annotation_filter)
        self.class_combo.clear(); self.class_combo.addItems(project.classes or ["object"])
        remembered_model = project_state.get("model", "")
        remembered_model_path = Path(remembered_model) if remembered_model else None
        if remembered_model_path and not remembered_model_path.is_absolute():
            remembered_model_path = project.root / remembered_model_path
        if remembered_model_path and remembered_model_path.is_file():
            try:
                self._activate_model(remembered_model_path, announce=False, persist=False)
            except Exception as exc:
                self.statusBar().showMessage(f"项目模型恢复失败：{exc}", 6000)
        self._restore_image_name = project_state.get("current_image", self._restore_image_name)
        self.refresh_list(); self.setWindowTitle(f"{APP_DISPLAY_NAME} — {project.root.name} [{project.dataset_format.upper()}]")
        if project.warnings and project.root not in self._shown_project_warning_roots:
            self._shown_project_warning_roots.add(project.root)
            details = "\n".join(project.warnings[:8])
            if len(project.warnings) > 8:
                details += f"\n……另有 {len(project.warnings) - 8} 条"
            QMessageBox.warning(self, "项目包含无法读取的文件", f"项目已打开，但有 {len(project.warnings)} 个文件被安全跳过：\n\n{details}")
        if project.items:
            restored = False
            if self._restore_image_name:
                for row in range(self.image_list.count()):
                    if Path(self.image_list.item(row).data(Qt.ItemDataRole.UserRole)).name == self._restore_image_name:
                        self.image_list.setCurrentRow(row); restored = True; break
            if not restored: self.image_list.setCurrentRow(0)
        remembered_class = project_state.get("current_class", self.settings.value("session/current_class", "", str))
        if remembered_class and self.class_combo.findText(remembered_class) >= 0:
            self.class_combo.setCurrentText(remembered_class)
        self.tabs.setCurrentIndex(int(project_state.get("tab", self.tabs.currentIndex())))

    @staticmethod
    def _load_project_ui_state(project):
        path = project.root / ".avls" / "ui_state.json"
        if not path.exists(): return {}
        try: return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError): return {}

    @staticmethod
    def _normalize_saved_queue(value):
        return normalize_saved_queue(value)

    def _save_project_ui_state(self):
        if not self.project: return
        path = self.project.root / ".avls" / "ui_state.json"; path.parent.mkdir(parents=True, exist_ok=True)
        model_value = ""
        if self.yolo.model_path:
            try: model_value = str(self.yolo.model_path.resolve().relative_to(self.project.root.resolve()))
            except ValueError: model_value = str(self.yolo.model_path.resolve())
        payload = {"current_image": self.current.path.name if self.current else "", "filter": self.filter_combo.currentText(), "annotation_filter": self.annotation_filter_combo.currentText(), "current_class": self.class_combo.currentText(), "tab": self.tabs.currentIndex(), "model": model_value}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def refresh_list(self):
        if not self.project: return
        selected_path = self.current.path if self.current else None
        selected_visible = False
        self.image_list.clear(); wanted = self.filter_combo.currentText(); annotation_wanted = self.annotation_filter_combo.currentText()
        for item in filtered_items(self.project, wanted, annotation_wanted):
            stage = self._queue_stage(item)
            prefix = {QUEUE_AUDIT:"①", QUEUE_FIX:"②", QUEUE_CONFIRM:"③", QUEUE_DONE:"✓", STAGE_UNLABELED:"○"}.get(stage, "·")
            short_stage = {QUEUE_AUDIT:"待检查", QUEUE_FIX:"待人工处理", QUEUE_CONFIRM:"待批量确认", QUEUE_DONE:"检查完成", STAGE_UNLABELED:"无标注"}.get(stage, stage)
            row = QListWidgetItem(f"{prefix} [{short_stage}] {item.path.name}  [{len(item.boxes)}]"); row.setData(Qt.ItemDataRole.UserRole, str(item.path)); self.image_list.addItem(row)
            if item.path == selected_path:
                self.image_list.setCurrentItem(row); selected_visible = True
        if not selected_visible:
            if self.current and self.current.dirty:
                save_item(self.project, self.current)
            self.current = None
            self.canvas.set_item(None)
            if self.image_list.count():
                self.image_list.setCurrentRow(0)
        self._refresh_issue_list()
        self._update_review_counts()

    def _update_review_counts(self):
        if not hasattr(self, "review_counts_label") or not self.project: return
        counts = queue_counts(self.project)
        unchecked = counts[QUEUE_AUDIT]; issues = counts[QUEUE_FIX]
        reviewed = counts[QUEUE_CONFIRM]; passed = counts[QUEUE_DONE]
        ai_predicted = sum(item.status == ImageStatus.AI_PREDICTED for item in self.project.items)
        self.review_counts_label.setText(
            f"① 待检查（模型/人工任选）：{unchecked} 张　　② 待人工处理：{issues} 张\n"
            f"③ 待批量确认：{reviewed} 张　　✓ 检查完成：{passed} 张\n"
            f"AI预测：{ai_predicted} 张"
        )

    def set_image_filter(self, text):
        index = self.filter_combo.findText(text)
        if index >= 0:
            self.filter_combo.setCurrentIndex(index)
        self.tabs.setCurrentIndex(0 if text == QUEUE_FIX else 2)

    def _queue_changed(self, text):
        self.refresh_list()
        if not hasattr(self, "queue_action_button"): return
        labels = {QUEUE_ALL:"请选择①②③工作队列", QUEUE_AUDIT:"选项 A：使用模型批量检查", QUEUE_FIX:"执行下一步：② 打开并人工处理", QUEUE_CONFIRM:"执行下一步：③ 批量确认这些图片", QUEUE_DONE:"✓ 已完成，无需继续处理"}
        help_texts = {
            QUEUE_ALL: "用于浏览全部图片。要执行批量操作，请选择下面的①、②或③队列。",
            QUEUE_AUDIT: "任选一种：A 用模型批量检查；B 人工查看当前图后按 V 通过；C 明确检查过左侧整批后批量通过。",
            QUEUE_FIX: "逐张处理模型发现的疑点；点击问题框后可修改类别、删除，或按 R 忽略误报。",
            QUEUE_CONFIRM: "问题已人工处理完毕；点击后把左侧这些图片一次性确认为质检完成。",
            QUEUE_DONE: "已经走完整个质检流程，只用于查看，不需要再次操作。",
        }
        self.queue_action_button.setText(labels.get(text, "请选择工作队列"))
        self.queue_action_button.setEnabled(text not in (QUEUE_ALL, QUEUE_DONE))
        self.manual_pass_current_button.setVisible(text == QUEUE_AUDIT)
        self.manual_pass_queue_button.setVisible(text == QUEUE_AUDIT)
        self.queue_help_label.setText(help_texts.get(text, "请选择工作队列。"))
        if hasattr(self, "tabs"):
            if text == QUEUE_FIX: self.tabs.setCurrentIndex(0)
            elif text in (QUEUE_AUDIT, QUEUE_CONFIRM): self.tabs.setCurrentIndex(2)

    def run_current_queue_action(self):
        queue = self.filter_combo.currentText()
        if queue == QUEUE_AUDIT:
            self.tabs.setCurrentIndex(2); self.audit_unchecked_images()
        elif queue == QUEUE_FIX:
            self.tabs.setCurrentIndex(0)
            if self.image_list.count() and self.image_list.currentRow() < 0: self.image_list.setCurrentRow(0)
            self.statusBar().showMessage("请点击问题框处理；R 表示原标注正确并忽略误报", 6000)
        elif queue == QUEUE_CONFIRM:
            self.tabs.setCurrentIndex(2); self.confirm_human_reviewed_images()
        else:
            QMessageBox.information(self, "请选择工作队列", "请先选择①待检查、②待人工处理或③待批量确认。")

    def confirm_left_queue_manually(self):
        """Mark the currently visible unchecked queue as human-reviewed without model QA."""
        if not self.project:
            return
        targets = self._visible_items_in_stage(QUEUE_AUDIT)
        if not targets:
            QMessageBox.information(self, "没有可人工通过的图片", "左侧当前范围内没有处于①待检查且包含标注框的图片。")
            return
        answer = QMessageBox.warning(
            self, "批量人工通过",
            f"将跳过模型质检，直接把左侧 {len(targets)} 张图片标记为“人工已检查通过”。\n\n"
            "请仅在你已经人工查看过这些图片时使用。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        for item in targets:
            for box in item.boxes:
                box.source = "human"; box.confidence = None
            clear_audit_report(self.project, item)
            item.audit_human_reviewed = True
            item.status = ImageStatus.CONFIRMED
            item.dirty = True
            save_item(self.project, item)
        self.canvas.update(); self.refresh_list()
        self.statusBar().showMessage(f"已跳过模型质检，人工通过左侧 {len(targets)} 张图片", 6000)

    def _visible_items_in_stage(self, stage):
        """Return only items currently shown on the left and belonging to stage."""
        if not self.project:
            return []
        paths = [Path(self.image_list.item(row).data(Qt.ItemDataRole.UserRole)) for row in range(self.image_list.count())]
        return items_from_visible_paths(self.project, paths, stage)

    def audit_unchecked_images(self):
        index = self.local_audit_scope.findData("unchecked")
        if index >= 0: self.local_audit_scope.setCurrentIndex(index)
        targets = self._visible_items_in_stage(QUEUE_AUDIT)
        self._audit_with_trained_model(targets, "左侧当前可见的待检查")

    @staticmethod
    def _has_been_audited(item):
        return bool(item.audit_passed or item.audit_issues or item.audit_human_reviewed)

    @staticmethod
    def _queue_stage(item):
        return stage_for(item)

    def _select_image(self, row):
        if row < 0 or not self.project: return
        entry = self.image_list.item(row); path = Path(entry.data(Qt.ItemDataRole.UserRole))
        if self.current and self.current.dirty: self.save_current()
        self.current = self.project.item_for_path(path); self.canvas.set_item(self.current)
        self.statusBar().showMessage(f"{path.name}  {self.current.width}×{self.current.height}  {len(self.current.boxes)} 个框")

    def _boxes_changed(self):
        if not self.current: return
        if self.current.audit_issues and self.project:
            # A direct canvas edit while reviewing is a human correction. Keep
            # a persistent reviewed marker instead of silently dropping the
            # image out of the review workflow.
            self.current.audit_issues = []
            self.current.audit_human_reviewed = True
            save_audit_report(self.project, self.current, {"ok": False, "summary": "标注框已人工调整，等待批量确认", "issues": []})
        elif self.current.audit_passed and self.project:
            # The annotation changed after its last audit. It must become
            # unchecked again so the next incremental audit can pick it up.
            clear_audit_report(self.project, self.current)
        elif self.current.audit_human_reviewed and self.current.status != ImageStatus.REVIEW_ISSUE:
            self.current.audit_human_reviewed = False
        if self.project:
            before = list(self.project.classes)
            self.project.update_classes(box.label for box in self.current.boxes)
            if self.project.classes != before: save_classes(self.project)
        if self.current.audit_human_reviewed: self.current.status = ImageStatus.REVIEW_ISSUE
        elif self.current.boxes and all(b.source == "ai" for b in self.current.boxes): self.current.status = ImageStatus.AI_PREDICTED
        elif self.current.boxes: self.current.status = ImageStatus.CONFIRMED
        else: self.current.status = ImageStatus.UNLABELED
        self._selection_changed(self.canvas.selected)

    def _selection_changed(self, index):
        if self.current and 0 <= index < len(self.current.boxes):
            b = self.current.boxes[index]; self.coords.setText(f"({b.xmin}, {b.ymin}) – ({b.xmax}, {b.ymax})\n来源: {b.source}")
            self.class_combo.setCurrentText(b.label)
        else: self.coords.setText("未选择标注框")

    def _audit_issue_selected(self, index):
        if not self.current or not (0 <= index < len(self.current.audit_issues)):
            self.issue_info.setText("未选择问题框")
            return
        issue = self.current.audit_issues[index]
        message = issue.get("message", "")
        suggested = issue.get("suggested_label")
        suggested_text = f"\n建议类别：{suggested}" if suggested else ""
        self.issue_info.setText(f"问题 {index + 1}：{issue.get('type', 'other')}\n{message}{suggested_text}")
        if suggested:
            if self.project and suggested not in self.project.classes:
                self.project.update_classes([suggested]); save_classes(self.project); self.class_combo.addItem(suggested)

    def apply_class_to_selected_issue(self):
        if not self.project or not self.current:
            return
        index = self.canvas.selected_audit_issue
        if not (0 <= index < len(self.current.audit_issues)):
            self.statusBar().showMessage("请先在画布中点击红色或黄色问题框", 3000)
            return
        previous_row = self.image_list.currentRow()
        issue_filter_active = self.filter_combo.currentText() == QUEUE_FIX
        label = self.class_combo.currentText().strip()
        if not label:
            self.error("请先选择或输入类别")
            return
        issue = self.current.audit_issues[index]
        box_index = issue.get("box_index")
        if isinstance(box_index, int) and 1 <= box_index <= len(self.current.boxes):
            box = self.current.boxes[box_index - 1]
            box.label = label; box.source = "human"; box.confidence = None
            action = "已修改现有标注框类别"
        else:
            bbox = issue.get("suggested_bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                self.error("该问题没有可用的建议框坐标，请手动画框")
                return
            try:
                box = AnnotationBox(label, *[round(float(value)) for value in bbox], source="human").normalized(self.current.width, self.current.height)
            except (TypeError, ValueError):
                self.error("建议框坐标无效")
                return
            if not box.valid:
                self.error("建议框范围无效")
                return
            self.current.boxes.append(box)
            action = "已把疑似漏标建议框添加为标注"
        self.project.update_classes([label]); save_classes(self.project)
        self.current.audit_issues.pop(index)
        self.current.audit_human_reviewed = True
        self.canvas.selected_audit_issue = -1
        if self.current.audit_issues:
            self.current.status = ImageStatus.REVIEW_ISSUE
            remaining_result = {
                "ok": False, "summary": f"人工处理后仍有 {len(self.current.audit_issues)} 个待处理问题", "issues": self.current.audit_issues,
            }
            save_audit_report(self.project, self.current, remaining_result)
            export_issue_image(self.project.root, self.current, remaining_result)
        else:
            self.current.status = ImageStatus.REVIEW_ISSUE
            save_audit_report(self.project, self.current, {"ok": False, "summary": "问题已人工修正，等待批量确认", "issues": []})
        self.current.dirty = True
        save_item(self.project, self.current)
        self.issue_info.setText("未选择问题框")
        resolved = self.current.audit_human_reviewed and not self.current.audit_issues
        self.canvas.update(); self.refresh_list()
        queue_complete = self._continue_issue_queue(previous_row, issue_filter_active and resolved)
        self._selection_changed(self.canvas.selected)
        suffix = "；质检问题已全部处理完成" if queue_complete else ""
        self.statusBar().showMessage(f"{action}：{label}{suffix}", 6000 if queue_complete else 4000)

    def apply_recommended_to_selected_issue(self):
        if not self.current:
            return
        index = self.canvas.selected_audit_issue
        if not (0 <= index < len(self.current.audit_issues)):
            self.statusBar().showMessage("请先点击问题框", 2500)
            return
        label = self.current.audit_issues[index].get("suggested_label")
        if not label:
            self.error("该问题没有推荐类别，请使用 Q/E、数字键或类别下拉框选择")
            return
        if self.project and label not in self.project.classes:
            self.project.update_classes([label]); save_classes(self.project); self.class_combo.addItem(label)
        self.class_combo.setCurrentText(label)
        self.apply_class_to_selected_issue()

    def delete_selected_issue(self):
        if not self.project or not self.current:
            return
        index = self.canvas.selected_audit_issue
        if not (0 <= index < len(self.current.audit_issues)):
            return
        previous_row = self.image_list.currentRow()
        issue_filter_active = self.filter_combo.currentText() == QUEUE_FIX
        issue = self.current.audit_issues[index]
        linked_box = issue.get("box_index")
        deleted_box = isinstance(linked_box, int) and 1 <= linked_box <= len(self.current.boxes)
        if deleted_box:
            self.current.boxes.pop(linked_box - 1)
        remaining = []
        for issue_index, other in enumerate(self.current.audit_issues):
            if issue_index == index:
                continue
            other_box = other.get("box_index")
            if deleted_box and other_box == linked_box:
                # Keep other findings independent. Their suggested bbox remains
                # actionable even though the original linked annotation is gone.
                other["box_index"] = None
            if deleted_box and isinstance(other_box, int) and other_box > linked_box:
                other["box_index"] = other_box - 1
            remaining.append(other)
        self.current.audit_issues = remaining
        self.current.audit_human_reviewed = True
        self.canvas.selected_audit_issue = -1
        self.canvas.selected = -1
        if remaining:
            self.current.status = ImageStatus.REVIEW_ISSUE
            remaining_result = {"ok": False, "summary": f"删除后仍有 {len(remaining)} 个待处理问题", "issues": remaining}
            save_audit_report(self.project, self.current, remaining_result)
            export_issue_image(self.project.root, self.current, remaining_result)
        else:
            self.current.status = ImageStatus.REVIEW_ISSUE
            save_audit_report(self.project, self.current, {"ok": False, "summary": "问题已人工修正，等待批量确认", "issues": []})
        self.current.dirty = True
        save_item(self.project, self.current)
        self.issue_info.setText("未选择问题框")
        resolved = self.current.audit_human_reviewed and not self.current.audit_issues
        self.canvas.selectionChanged.emit(-1)
        self.canvas.update(); self.refresh_list()
        queue_complete = self._continue_issue_queue(previous_row, issue_filter_active and resolved)
        message = "已删除关联标注框" if deleted_box else "已驳回并移除漏标建议"
        if queue_complete: message += "；质检问题已全部处理完成"
        self.statusBar().showMessage(message, 6000 if queue_complete else 4000)

    def ignore_selected_issue(self):
        """Keep the annotation and remember this model finding as a false alarm."""
        if not self.project or not self.current:
            return
        index = self.canvas.selected_audit_issue
        if not (0 <= index < len(self.current.audit_issues)):
            self.statusBar().showMessage("请先点击要忽略的问题框", 2500)
            return
        previous_row = self.image_list.currentRow()
        issue_filter_active = self.filter_combo.currentText() == QUEUE_FIX
        issue = self.current.audit_issues[index]
        ignore_audit_issue(self.project, self.current, issue)
        self.current.audit_issues.pop(index)
        self.current.audit_human_reviewed = True
        self.canvas.selected_audit_issue = -1
        if self.current.audit_issues:
            self.current.status = ImageStatus.REVIEW_ISSUE
            result = {"ok": False, "summary": f"已忽略误报，仍有 {len(self.current.audit_issues)} 个待处理问题", "issues": self.current.audit_issues}
        else:
            self.current.status = ImageStatus.REVIEW_ISSUE
            result = {"ok": False, "summary": "所有质检提示已处理，等待批量确认", "issues": []}
        save_audit_report(self.project, self.current, result)
        save_item(self.project, self.current)
        resolved = not self.current.audit_issues
        self.issue_info.setText("未选择问题框")
        self.canvas.update(); self.refresh_list()
        queue_complete = self._continue_issue_queue(previous_row, issue_filter_active and resolved)
        message = "已保留原标注并记住该误报；重新审查时不会重复提示"
        if queue_complete: message += "；当前待修正队列已处理完成"
        self.statusBar().showMessage(message, 6000)

    def _continue_issue_queue(self, previous_row, should_advance):
        if not should_advance:
            return False
        if self.image_list.count():
            # Prefer the next item which still has unresolved issues. Reviewed
            # items remain visible in the queue with a check mark until the
            # user confirms them in one batch.
            start = min(max(previous_row, 0), self.image_list.count() - 1)
            for offset in range(self.image_list.count()):
                row = (start + offset) % self.image_list.count()
                path = Path(self.image_list.item(row).data(Qt.ItemDataRole.UserRole))
                item = self.project.item_for_path(path)
                if item and item.audit_issues:
                    self.image_list.setCurrentRow(row)
                    return False
            self.image_list.setCurrentRow(start)
            return True
        else:
            return True

    def confirm_human_reviewed_images(self):
        if not self.project:
            return
        targets = self._visible_items_in_stage(QUEUE_CONFIRM)
        if not targets:
            QMessageBox.information(
                self, "左侧没有可确认图片",
                "左侧当前范围内没有处于③“待批量确认”的图片，因此不会执行任何操作。\n\n"
                "如需确认图片，请先选择③工作队列，并检查第二层“标注状态”筛选是否过窄。",
            )
            return
        answer = QMessageBox.question(
            self, "确认人工已复核图片",
            f"将把 {len(targets)} 张带 ☑ 标记且已无待处理问题的图片设为“人工确认”。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        for item in targets:
            for box in item.boxes:
                box.source = "human"; box.confidence = None
            item.status = ImageStatus.CONFIRMED
            item.dirty = True
            clear_audit_report(self.project, item)
            item.audit_human_reviewed = True
            save_item(self.project, item)
        self.canvas.update(); self.refresh_list()
        self.statusBar().showMessage(f"已人工确认 {len(targets)} 张复核标记图片", 6000)

    def _class_changed(self, label):
        self.canvas.current_label = label.strip() or "object"
        if self.current and 0 <= self.canvas.selected < len(self.current.boxes) and label.strip():
            box = self.current.boxes[self.canvas.selected]
            if box.label != label.strip():
                if self.current.audit_passed and self.project:
                    clear_audit_report(self.project, self.current)
                elif self.current.audit_human_reviewed and self.current.status != ImageStatus.REVIEW_ISSUE:
                    self.current.audit_human_reviewed = False
                box.label = label.strip(); box.source = "human"; box.confidence = None
                self.current.status = ImageStatus.CONFIRMED; self.current.dirty = True
                self.canvas.update(); self._selection_changed(self.canvas.selected)

    def cycle_selected_class(self, direction):
        has_box = self.current and 0 <= self.canvas.selected < len(self.current.boxes)
        has_issue = self.current and 0 <= self.canvas.selected_audit_issue < len(self.current.audit_issues)
        if not self.project or not self.current or not (has_box or has_issue):
            self.statusBar().showMessage("请先在画布中选择一个标注框", 2500)
            return
        classes = self.project.classes
        if not classes:
            self.statusBar().showMessage("请先添加类别", 2500)
            return
        current_label = self.current.boxes[self.canvas.selected].label if has_box else self.class_combo.currentText().strip()
        try: index = classes.index(current_label)
        except ValueError: index = 0 if direction < 0 else -1
        self.set_selected_class_by_index((index + direction) % len(classes), apply_issue=True)

    def set_selected_class_by_index(self, index, apply_issue=True):
        has_box = self.current and 0 <= self.canvas.selected < len(self.current.boxes)
        has_issue = self.current and 0 <= self.canvas.selected_audit_issue < len(self.current.audit_issues)
        if not self.project or not self.current or not (has_box or has_issue):
            return
        if not (0 <= index < len(self.project.classes)):
            self.statusBar().showMessage(f"类别 {index + 1} 尚未创建", 2000)
            return
        label = self.project.classes[index]
        self.class_combo.setCurrentText(label)
        self._class_changed(label)
        if has_issue and apply_issue:
            self.apply_class_to_selected_issue()
            self.statusBar().showMessage(f"问题框已应用类别 {index + 1}：{label}", 2500)
        elif has_issue:
            self.statusBar().showMessage(f"问题框待应用类别：{label}（按 Enter 确认）", 3000)
        else:
            self.statusBar().showMessage(f"已切换为类别 {index + 1}：{label}", 2000)

    def add_class(self):
        label, ok = QInputDialog.getText(self, "添加类别", "类别名称")
        if ok and label.strip() and self.project:
            self.project.update_classes([label]); save_classes(self.project)
            if self.class_combo.findText(label.strip()) < 0: self.class_combo.addItem(label.strip())
            self.class_combo.setCurrentText(label.strip())

    def delete_current_class(self):
        if not self.project:
            return
        label = self.class_combo.currentText().strip()
        if not label or label not in self.project.classes:
            self.statusBar().showMessage("请选择一个已存在的类别", 2500)
            return
        affected_items = []
        affected_boxes = 0
        for item in self.project.items:
            count = sum(box.label == label for box in item.boxes)
            if count:
                affected_items.append(item)
                affected_boxes += count
        message = f"确定删除类别“{label}”吗？"
        if affected_boxes:
            message += f"\n\n整个项目中有 {affected_boxes} 个框使用该类别，分布在 {len(affected_items)} 张图片中；这些框也会被删除。"
        message += "\n\n此操作不会删除图片。"
        answer = QMessageBox.warning(self, "删除类别", message, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        for item in affected_items:
            item.boxes = [box for box in item.boxes if box.label != label]
            item.status = ImageStatus.CONFIRMED if item.boxes else ImageStatus.UNLABELED
            item.dirty = True
            save_item(self.project, item)
        self.project.classes.remove(label)
        save_classes(self.project)
        self.class_combo.blockSignals(True)
        self.class_combo.clear(); self.class_combo.addItems(self.project.classes)
        self.class_combo.blockSignals(False)
        self.canvas.selected = -1
        self.canvas.current_label = self.class_combo.currentText().strip() or "object"
        self.canvas.update(); self._selection_changed(-1); self.refresh_list()
        self.statusBar().showMessage(f"已删除类别：{label}（同时删除 {affected_boxes} 个框）", 5000)

    def save_current(self):
        if self.project and self.current:
            path = save_item(self.project, self.current); self.statusBar().showMessage(f"已保存 {path.name}", 3000)

    def confirm_current_annotations(self):
        if not self.project or not self.current:
            return
        if not self.current.boxes:
            self.statusBar().showMessage("当前图片没有标注框", 2500)
            return
        previous_row = self.image_list.currentRow()
        continue_manual_queue = self.filter_combo.currentText() == QUEUE_AUDIT
        changed = 0
        for box in self.current.boxes:
            if box.source == "ai":
                box.source = "human"
                box.confidence = None
                changed += 1
        self.current.status = ImageStatus.CONFIRMED
        self.current.dirty = True
        clear_audit_report(self.project, self.current)
        self.current.audit_human_reviewed = True
        save_item(self.project, self.current)
        self.canvas.update(); self.refresh_list()
        if continue_manual_queue and self.image_list.count():
            self.image_list.setCurrentRow(min(max(previous_row, 0), self.image_list.count() - 1))
        self.statusBar().showMessage(f"当前图片已人工确认，共确认 {changed} 个 AI 框", 4000)

    def navigate(self, delta):
        row = self.image_list.currentRow(); target = row + delta
        if 0 <= target < self.image_list.count(): self.image_list.setCurrentRow(target)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载检测模型", "", "检测模型 (*.pt *.onnx *.tflite *.engine *.torchscript *.mlpackage *.pb);;所有文件 (*)")
        if not path: return
        try:
            self._activate_model(Path(path))
        except Exception as exc: self.error(str(exc))

    def _activate_model(self, path: Path, announce=True, persist=True):
        path = path.resolve()
        self.yolo.load(path, self.project.classes if self.project else [])
        self.model_label.setText(f"{path.name}\n{self.yolo.model_description}")
        if self.project:
            self.project.model_path = path
            if persist:
                self._save_project_ui_state()
        self.settings.setValue("ai/model", str(path)); self.settings.sync()
        if announce:
            self.statusBar().showMessage(f"模型已加载：{self.yolo.model_description}", 6000)

    def predict_current(self):
        if not self.current: return
        self._run(lambda: self.yolo.predict(self.current.path, self.confidence.value()), self._apply_prediction, "正在推理当前图片…")

    def _apply_prediction(self, boxes):
        self.current.boxes = boxes; self.current.dirty = True; self.current.status = ImageStatus.AI_PREDICTED if boxes else ImageStatus.UNLABELED
        if self.project: self.project.update_classes(b.label for b in boxes); save_item(self.project, self.current)
        self.canvas.update(); self.refresh_list(); self.statusBar().showMessage(f"AI 生成 {len(boxes)} 个框", 4000)

    def predict_batch(self):
        if not self.project: return
        targets = [i for i in self.project.items if i.status == ImageStatus.UNLABELED]
        def job():
            for item in targets:
                item.boxes = self.yolo.predict(item.path, self.confidence.value()); item.status = ImageStatus.AI_PREDICTED if item.boxes else ImageStatus.UNLABELED; save_item(self.project, item)
            return len(targets)
        self._run(job, lambda n: (self.refresh_list(), self.statusBar().showMessage(f"批量处理完成：{n} 张", 5000)), "正在批量自动标注…")

    def quick_train(self):
        if not self.project: return
        epochs, ok = QInputDialog.getInt(self, "快速训练", "Epochs", 30, 1, 500)
        if not ok: return
        base_model = self.training_base.currentData()
        def job():
            if not self.yolo.can_train:
                self.yolo.load(Path(base_model))
            return self.yolo.train_voc_project(self.project, epochs)
        def done(path):
            self.model_label.setText(Path(path).name)
            if self.project: self.project.model_path = Path(path)
            self.settings.setValue("ai/model", str(Path(path).resolve())); self.settings.sync()
            self._save_project_ui_state()
            self.statusBar().showMessage(f"训练完成：{path}", 8000)
        initial = f"正在下载并加载 {base_model}…" if self.yolo.model is None else "正在训练，请查看终端日志…"
        self._run(job, done, initial)

    @staticmethod
    def _training_device_text():
        try:
            import torch
            if torch.cuda.is_available():
                return f"GPU 0：{torch.cuda.get_device_name(0)}"
            return "CPU（未检测到 CUDA）"
        except Exception:
            return "CPU（PyTorch 不可用）"

    def _load_provider_config(self):
        self.provider_name.setText(self.settings.value("vision/name", "OpenAI", str))
        self.provider_base_url.setText(self.settings.value("vision/base_url", "https://api.openai.com/v1", str))
        self.provider_model.setText(self.settings.value("vision/model", "gpt-4o-mini", str))
        protocol = self.settings.value("vision/protocol", "responses", str)
        index = self.provider_protocol.findData(protocol); self.provider_protocol.setCurrentIndex(max(0, index))
        self.provider_key_env.setText(self.settings.value("vision/key_env", "OPENAI_API_KEY", str))

    def save_provider_config(self):
        self.settings.setValue("vision/name", self.provider_name.text().strip())
        self.settings.setValue("vision/base_url", self.provider_base_url.text().strip())
        self.settings.setValue("vision/model", self.provider_model.text().strip())
        self.settings.setValue("vision/protocol", self.provider_protocol.currentData())
        self.settings.setValue("vision/key_env", self.provider_key_env.text().strip())
        self.settings.sync()
        self.statusBar().showMessage("模型 API 配置已保存（API Key 未保存）", 4000)

    def _provider_config(self):
        base_url = self.provider_base_url.text().strip()
        model = self.provider_model.text().strip()
        if not base_url or not model:
            raise RuntimeError("请填写 Base URL 和模型名称")
        return VisionProviderConfig(
            name=self.provider_name.text().strip() or "Vision API", base_url=base_url, model=model,
            protocol=self.provider_protocol.currentData(), api_key_env=self.provider_key_env.text().strip() or "OPENAI_API_KEY",
            api_key=self.provider_api_key.text(),
        )

    def audit_current(self):
        if not self.current or not self.project:
            self.error("请先打开项目并选择图片"); return
        if not self.current.boxes:
            self.error("当前图片没有标注框，无法进行完成后质检"); return
        try: config = self._provider_config()
        except Exception as exc: self.error(str(exc)); return
        self._run(lambda: self.vision_audit.audit(self.current, self.project.classes, config), self._show_audit_result, "正在进行视觉标注质检…")

    def audit_all_with_trained_model(self):
        if not self.project:
            return
        include_all = self.local_audit_scope.currentData() == "all"
        targets = [item for item in self.project.items if item.boxes and (include_all or not self._has_been_audited(item))]
        scope_text = "全部已标注" if include_all else "全项目尚未检查的"
        self._audit_with_trained_model(targets, scope_text)

    def _audit_with_trained_model(self, targets, scope_text):
        if not self.project:
            return
        if self.yolo.model is None:
            self.error("请先训练模型或加载检测模型")
            return
        if not targets:
            QMessageBox.information(self, "当前范围没有待检查图片", "当前操作范围为空，因此不会运行模型，也不会处理隐藏图片。\n\n请检查左侧工作队列和第二层标注状态筛选。")
            return
        answer = QMessageBox.question(
            self, "训练模型批量审查",
            f"将使用当前模型“{self.model_label.text()}”审查 {len(targets)} 张{scope_text}图片，并按类别和 IoU 比较。\n\n此功能不消耗 API/Codex Token。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        project = self.project
        policy = self._audit_policy()
        def job():
            reports = []
            for item in targets:
                result = self.yolo.audit_annotations(item, **policy)
                result["issues"] = filter_ignored_audit_issues(project, item, result.get("issues", [])); result["ok"] = not result["issues"]
                issues = result.get("issues", [])
                if issues:
                    item.status = ImageStatus.REVIEW_ISSUE
                    save_audit_report(project, item, result)
                    output = export_issue_image(project.root, item, result)
                else:
                    clear_audit_report(project, item)
                    save_audit_report(project, item, result)
                    output = None
                reports.append((item.path.name, result, str(output) if output else ""))
            return reports
        self._run(job, self._show_trained_model_batch_result, "正在用当前模型批量审查图片…")

    def confirm_audited_safe_ai(self):
        if not self.project:
            return
        targets = [item for item in self.project.items if item.status == ImageStatus.AI_PREDICTED]
        if not targets:
            QMessageBox.information(self, "没有 AI预测图片", "当前项目中没有状态为 AI预测的图片。")
            return
        already_safe = [item for item in targets if item.audit_passed and not item.audit_issues]
        need_audit = [item for item in targets if item not in already_safe]
        if need_audit and self.yolo.model is None:
            self.error("存在尚未质检的 AI预测图片，请先训练模型或加载 best.pt。软件将使用该模型自动审查后再确认。")
            return
        answer = QMessageBox.question(
            self, "一键审查并确认 AI预测",
            f"AI预测图片：{len(targets)} 张\n已质检通过：{len(already_safe)} 张\n需要自动质检：{len(need_audit)} 张\n\n软件将使用当前 best.pt 和 GPU 审查未质检图片；无疑点的自动确认，有疑点的转入“质检有问题”。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        project = self.project
        policy = self._audit_policy()
        def job():
            confirmed = 0
            problem = 0
            for item in targets:
                if item.audit_passed and not item.audit_issues:
                    result = {"ok": True, "issues": []}
                else:
                    result = self.yolo.audit_annotations(item, **policy)
                    result["issues"] = filter_ignored_audit_issues(project, item, result.get("issues", [])); result["ok"] = not result["issues"]
                issues = result.get("issues", [])
                if issues:
                    item.status = ImageStatus.REVIEW_ISSUE
                    save_audit_report(project, item, result)
                    export_issue_image(project.root, item, result)
                    problem += 1
                else:
                    for box in item.boxes:
                        if box.source == "ai":
                            box.source = "human"; box.confidence = None
                    item.status = ImageStatus.CONFIRMED
                    clear_audit_report(project, item)
                    save_item(project, item)
                    save_audit_report(project, item, {"ok": True, "issues": [], "summary": "训练模型审查通过并自动确认"})
                    confirmed += 1
            return confirmed, problem
        def done(result):
            confirmed, problem = result
            self.refresh_list(); self.canvas.update()
            QMessageBox.information(self, "一键处理完成", f"自动确认：{confirmed} 张\n转入质检有问题：{problem} 张")
        self._run(job, done, "正在用训练模型审查并确认 AI预测图片…")

    def _audit_policy(self, profile=None):
        profile = profile or self.audit_profile.currentData()
        base = self.confidence.value()
        if profile == "conservative":
            return dict(confidence=max(base, 0.50), iou_threshold=0.30, missing_confidence=0.85,
                        class_confidence=0.85, class_mismatch_iou=0.45,
                        report_unmatched=False, ignore_edge_missing=True)
        if profile == "balanced":
            return dict(confidence=max(base, 0.35), iou_threshold=0.42, missing_confidence=0.68,
                        class_confidence=0.72, class_mismatch_iou=0.32,
                        report_unmatched=True, ignore_edge_missing=True)
        return dict(confidence=base, iou_threshold=0.50, missing_confidence=max(base, 0.45),
                    class_confidence=max(base, 0.55), class_mismatch_iou=0.20,
                    report_unmatched=True, ignore_edge_missing=False)

    def rescreen_existing_issues(self):
        if not self.project: return
        if self.yolo.model is None:
            self.error("请先加载用于质检的检测模型")
            return
        targets = [item for item in self.project.items if item.audit_issues]
        if not targets:
            QMessageBox.information(self, "没有待重筛图片", "当前没有包含质检问题的图片。")
            return
        answer = QMessageBox.question(
            self, "保守重筛问题队列",
            f"将用保守规则重新审查 {len(targets)} 张问题图片：忽略模型漏检、低于85%的漏标候选、贴边候选和低可靠错类。\n\n"
            "清理后无剩余问题的图片会进入 [③ 待批量确认]，之后可用③一次通过。不会修改任何标注框。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes: return
        project = self.project; policy = self._audit_policy("conservative")
        def job():
            cleared = 0; remaining = 0; removed = 0
            for item in targets:
                old_count = len(item.audit_issues)
                result = self.yolo.audit_annotations(item, **policy)
                result["issues"] = filter_ignored_audit_issues(project, item, result.get("issues", [])); result["ok"] = not result["issues"]
                removed += max(0, old_count-len(result["issues"]))
                item.audit_human_reviewed = not result["issues"]
                item.status = ImageStatus.REVIEW_ISSUE
                if result["issues"]:
                    save_audit_report(project, item, result); export_issue_image(project.root, item, result); remaining += 1
                else:
                    save_audit_report(project, item, {"ok": False, "summary": "保守重筛未发现高可信问题，等待批量确认", "issues": []}); cleared += 1
                save_item(project, item)
            return cleared, remaining, removed
        def done(values):
            cleared, remaining, removed = values; self.refresh_list(); self.canvas.update()
            QMessageBox.information(self, "重筛完成", f"自动清理低可靠提示：{removed} 个\n转入 [③ 待批量确认]：{cleared} 张\n仍需人工处理：{remaining} 张\n\n现在可在左侧选择③并批量确认。")
        self._run(job, done, "正在用保守规则重筛现有问题队列…")

    def _show_trained_model_batch_result(self, reports):
        clean = sum(not result.get("issues", []) for _, result, _ in reports)
        problem = len(reports) - clean
        blocks = []
        for name, result, output in reports:
            block = self._format_audit_result(name, result)
            if output:
                block += f"\n问题定位图：{output}"
            blocks.append(block)
        self.gpt_output.setPlainText("\n\n".join(blocks))
        self.refresh_list(); self.canvas.update()
        self.statusBar().showMessage(f"训练模型审查完成：无明显差异 {clean} 张，发现疑点 {problem} 张", 10000)
        QMessageBox.information(
            self, "训练模型审查完成",
            f"无明显差异：{clean} 张\n发现疑点：{problem} 张\n\n该结果表示模型与标注不一致，不代表人工标注一定错误。\n问题文件：\n{self.project.root / '.avls' / 'review_issues'}",
        )

    def choose_model_audit_scope(self):
        choice, ok = QInputDialog.getItem(
            self, "选择模型审查范围", "审查范围",
            ["仅未确认图片（推荐）", "所有已标注图片"], 0, False,
        )
        if ok:
            self.audit_unconfirmed_with_model(include_confirmed=choice == "所有已标注图片")

    def audit_unconfirmed_with_model(self, include_confirmed=False):
        if not self.project:
            return
        if include_confirmed:
            targets = [item for item in self.project.items if item.boxes]
            scope_text = "所有已标注图片"
        else:
            targets = [item for item in self.project.items if item.boxes and item.status in (ImageStatus.AI_PREDICTED, ImageStatus.REVIEW_ISSUE)]
            scope_text = "未确认图片"
        if not targets:
            self.error(f"没有可审查的{scope_text}")
            return
        try:
            config = self._provider_config()
        except Exception as exc:
            self.error(str(exc)); return
        answer = QMessageBox.question(
            self, "模型批量审查",
            f"将使用“{config.name} / {config.model}”依次审查 {len(targets)} 张{scope_text}，每张图片会调用一次模型接口，可能产生费用。\n\n审查通过的图片将保持或转为人工确认；有问题的图片会输出定位图和 JSON 报告。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        project = self.project
        classes = list(project.classes)
        def job():
            reports = []
            for item in targets:
                result = self.vision_audit.audit(item, classes, config)
                issues = result.get("issues", [])
                if result.get("ok", not issues) and not issues:
                    for box in item.boxes:
                        if box.source == "ai":
                            box.source = "human"
                            box.confidence = None
                    item.status = ImageStatus.CONFIRMED
                    clear_audit_report(project, item)
                    save_item(project, item)
                    save_audit_report(project, item, result)
                    output = None
                else:
                    item.status = ImageStatus.REVIEW_ISSUE
                    save_audit_report(project, item, result)
                    output = export_issue_image(project.root, item, result)
                reports.append((item.path.name, result, str(output) if output else ""))
            return reports
        self._run(job, lambda reports: self._show_review_batch_result(reports, "模型"), f"正在使用 {config.name} 批量审查{scope_text}…")

    def audit_current_with_codex(self):
        if not self.current or not self.project:
            self.error("请先打开项目并选择图片"); return
        if not self.current.boxes:
            self.error("当前图片没有标注框，无法进行完成后质检"); return
        self._run(lambda: self.vision_audit.audit_with_codex(self.current, self.project.classes), self._show_audit_result, "正在调用本机 Codex 进行标注质检…")

    def audit_and_confirm_with_codex(self):
        if not self.current or not self.project:
            self.error("请先打开项目并选择图片"); return
        if not self.current.boxes:
            self.error("当前图片没有标注框，无法进行完成后质检"); return
        audited_item = self.current
        def done(result):
            self.gpt_output.setPlainText(self._format_audit_result(audited_item.path.name, result))
            issues = result.get("issues", [])
            if result.get("ok", not issues) and not issues and self.current is audited_item:
                self.confirm_current_annotations()
                QMessageBox.information(self, "Codex 审查通过", "Codex 未发现明显标注问题，当前图片已自动确认并保存。")
            else:
                audited_item.status = ImageStatus.REVIEW_ISSUE
                save_audit_report(self.project, audited_item, result)
                output = export_issue_image(self.project.root, audited_item, result)
                if output:
                    self.gpt_output.appendPlainText(f"\n问题定位图：{output}")
                self.refresh_list(); self.canvas.update()
                QMessageBox.warning(self, "Codex 发现疑似问题", f"发现 {len(issues)} 个疑似问题，当前图片未自动确认。请查看质检报告并人工修正。")
        self._run(lambda: self.vision_audit.audit_with_codex(audited_item, self.project.classes), done, "正在调用本机 Codex 审查；通过后将自动确认…")

    def audit_unconfirmed_with_codex(self):
        if not self.project:
            return
        targets = [item for item in self.project.items if item.boxes and item.status in (ImageStatus.AI_PREDICTED, ImageStatus.REVIEW_ISSUE)]
        if not targets:
            self.error("没有需要审查的 AI预测或质检有问题图片")
            return
        answer = QMessageBox.question(
            self, "本机 Codex 批量审查",
            f"将依次审查 {len(targets)} 张未确认图片，会消耗 Codex 套餐额度。\n\n审查通过的图片将自动确认；有问题的图片会输出定位图和 JSON 报告。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        project = self.project
        classes = list(project.classes)
        def job():
            reports = []
            for item in targets:
                result = self.vision_audit.audit_with_codex(item, classes)
                issues = result.get("issues", [])
                if result.get("ok", not issues) and not issues:
                    for box in item.boxes:
                        if box.source == "ai":
                            box.source = "human"
                            box.confidence = None
                    item.status = ImageStatus.CONFIRMED
                    clear_audit_report(project, item)
                    save_item(project, item)
                    save_audit_report(project, item, result)
                    output = None
                else:
                    item.status = ImageStatus.REVIEW_ISSUE
                    save_audit_report(project, item, result)
                    output = export_issue_image(project.root, item, result)
                reports.append((item.path.name, result, str(output) if output else ""))
            return reports
        self._run(job, lambda reports: self._show_review_batch_result(reports, "Codex"), "正在用本机 Codex 批量审查未确认图片…")

    def _show_review_batch_result(self, reports, source):
        passed = sum(not result.get("issues", []) and result.get("ok", True) for _, result, _ in reports)
        problem = len(reports) - passed
        blocks = []
        for name, result, output in reports:
            block = self._format_audit_result(name, result)
            if output:
                block += f"\n问题定位图：{output}"
            blocks.append(block)
        self.gpt_output.setPlainText("\n\n".join(blocks))
        self.refresh_list(); self.canvas.update()
        self.statusBar().showMessage(f"{source} 批量审查完成：通过并确认 {passed} 张，发现问题 {problem} 张", 10000)
        QMessageBox.information(self, f"{source} 批量审查完成", f"通过并自动确认：{passed} 张\n发现疑似问题：{problem} 张\n\n问题文件已输出到：\n{self.project.root / '.avls' / 'review_issues'}")

    def audit_all_labeled(self):
        if not self.project: return
        items = [item for item in self.project.items if item.boxes]
        if not items:
            self.error("项目中没有已标注图片"); return
        answer = QMessageBox.question(self, "批量 API 质检", f"将对 {len(items)} 张图片分别调用一次模型 API，可能产生费用。是否继续？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes: return
        try: config = self._provider_config()
        except Exception as exc: self.error(str(exc)); return
        def job():
            reports = []
            for item in items:
                reports.append((item.path.name, self.vision_audit.audit(item, self.project.classes, config)))
            return reports
        self._run(job, self._show_batch_audit_result, "正在批量进行视觉标注质检…")

    def _show_audit_result(self, result):
        if self.project and self.current:
            issues = result.get("issues", [])
            if issues:
                self.current.status = ImageStatus.REVIEW_ISSUE
                save_audit_report(self.project, self.current, result)
                output = export_issue_image(self.project.root, self.current, result)
            else:
                clear_audit_report(self.project, self.current)
                save_audit_report(self.project, self.current, result)
                output = None
            text = self._format_audit_result(self.current.path.name, result)
            if output: text += f"\n\n问题定位图：{output}"
            self.gpt_output.setPlainText(text)
            self.refresh_list(); self.canvas.update()
        else:
            self.gpt_output.setPlainText(self._format_audit_result("", result))
        count = len(result.get("issues", [])); self.statusBar().showMessage(f"质检完成：发现 {count} 个疑似问题", 6000)

    def _show_batch_audit_result(self, reports):
        text = "\n\n".join(self._format_audit_result(name, result) for name, result in reports)
        self.gpt_output.setPlainText(text)
        total = sum(len(result.get("issues", [])) for _, result in reports)
        self.statusBar().showMessage(f"批量质检完成：发现 {total} 个疑似问题", 8000)

    def _refresh_issue_list(self):
        if not hasattr(self, "audit_issue_list"):
            return
        self.audit_issue_list.clear()
        if not self.project:
            return
        for image in self.project.items:
            for issue_index, issue in enumerate(image.audit_issues):
                severity = issue.get("severity", "unknown")
                issue_type = issue.get("type", "other")
                row = QListWidgetItem(f"[{severity}] {image.path.name} · {issue_type}")
                row.setData(Qt.ItemDataRole.UserRole, str(image.path))
                row.setData(Qt.ItemDataRole.UserRole + 1, issue_index)
                self.audit_issue_list.addItem(row)

    def _jump_to_audit_issue(self, row):
        if not self.project:
            return
        path = Path(row.data(Qt.ItemDataRole.UserRole))
        issue_index = int(row.data(Qt.ItemDataRole.UserRole + 1) or 0)
        self.filter_combo.setCurrentText(QUEUE_FIX)
        self.refresh_list()
        for index in range(self.image_list.count()):
            if Path(self.image_list.item(index).data(Qt.ItemDataRole.UserRole)) == path:
                self.image_list.setCurrentRow(index)
                break
        if self.current and issue_index < len(self.current.audit_issues):
            box_index = self.current.audit_issues[issue_index].get("box_index")
            if isinstance(box_index, int) and 1 <= box_index <= len(self.current.boxes):
                self.canvas.selected = box_index - 1
                self.canvas.selectionChanged.emit(self.canvas.selected)
            self.canvas.update()

    @staticmethod
    def _format_audit_result(name, result):
        issues = result.get("issues", [])
        lines = [f"图片：{name}", f"结论：{'未发现明显问题' if not issues else f'发现 {len(issues)} 个疑似问题'}", f"摘要：{result.get('summary', '')}"]
        for index, issue in enumerate(issues, 1):
            box = issue.get("box_index"); box_text = f"框 #{box}" if box is not None else "可能漏标"
            lines.append(f"\n{index}. [{issue.get('severity', 'unknown')}] {issue.get('type', 'other')} · {box_text}\n   {issue.get('message', '')}")
            if issue.get("suggested_label"): lines.append(f"   建议类别：{issue['suggested_label']}")
            if issue.get("suggested_bbox"): lines.append(f"   建议框：{issue['suggested_bbox']}")
        lines.append("\n注意：视觉模型质检属于辅助判断，最终结果需要人工确认。")
        return "\n".join(lines)

    def _run(self, fn, done, message):
        if self.worker and self.worker.isRunning(): self.error("已有任务正在运行"); return
        self.setEnabled(False); self.statusBar().showMessage(message)
        self.worker = TaskThread(fn); self.worker.done.connect(lambda value: (self.setEnabled(True), done(value))); self.worker.failed.connect(lambda msg: (self.setEnabled(True), self.error(msg))); self.worker.start()

    def error(self, message): QMessageBox.critical(self, "错误", message)

    def _restore_window_state(self):
        geometry = self.settings.value("window/geometry")
        if geometry is not None: self.restoreGeometry(geometry)
        splitter_state = self.settings.value("window/splitter")
        if splitter_state is not None: self.splitter.restoreState(splitter_state)
        remembered_queue = self._normalize_saved_queue(self.settings.value("session/filter", QUEUE_ALL, str))
        if self.filter_combo.findText(remembered_queue) >= 0:
            self.filter_combo.setCurrentText(remembered_queue)
        remembered_annotation_filter = self.settings.value("session/annotation_filter", "全部标注状态", str)
        if self.annotation_filter_combo.findText(remembered_annotation_filter) >= 0:
            self.annotation_filter_combo.setCurrentText(remembered_annotation_filter)
        self.confidence.setValue(self.settings.value("ai/confidence", 0.25, float))
        base_model = self.settings.value("ai/training_base", "yolo11n.pt", str)
        base_index = self.training_base.findData(base_model)
        if base_index >= 0: self.training_base.setCurrentIndex(base_index)
        crosshair = self.settings.value("canvas/crosshair", True, bool)
        self.crosshair_action.setChecked(crosshair); self.canvas.set_crosshair_enabled(crosshair)
        self._restore_image_name = self.settings.value("session/current_image", "", str)
        self.tabs.setCurrentIndex(self.settings.value("session/tab", 0, int))
        scope = self.settings.value("review/local_scope", "unchecked", str)
        scope_index = self.local_audit_scope.findData(scope)
        if scope_index >= 0: self.local_audit_scope.setCurrentIndex(scope_index)
        profile = self.settings.value("review/profile", "conservative", str)
        profile_index = self.audit_profile.findData(profile)
        if profile_index >= 0: self.audit_profile.setCurrentIndex(profile_index)
        self._queue_changed(self.filter_combo.currentText())

    def restore_last_project(self):
        path_text = self.settings.value("session/project", "", str)
        if path_text:
            path = Path(path_text)
            if path.is_dir():
                self.open_project(path)
        model_text = self.settings.value("ai/model", "", str)
        if self.yolo.model is None and model_text and Path(model_text).is_file():
            try:
                self._activate_model(Path(model_text), announce=False)
            except Exception as exc:
                self.statusBar().showMessage(f"上次模型恢复失败：{exc}", 6000)

    def _save_session(self):
        self._save_project_ui_state()
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/splitter", self.splitter.saveState())
        self.settings.setValue("session/filter", self.filter_combo.currentText())
        self.settings.setValue("session/annotation_filter", self.annotation_filter_combo.currentText())
        self.settings.setValue("session/current_class", self.class_combo.currentText())
        self.settings.setValue("session/current_image", self.current.path.name if self.current else "")
        self.settings.setValue("session/project", str(self.project.root) if self.project else "")
        self.settings.setValue("canvas/crosshair", self.crosshair_action.isChecked())
        self.settings.setValue("ai/confidence", self.confidence.value())
        self.settings.setValue("ai/training_base", self.training_base.currentData())
        self.settings.setValue("ai/model", str(self.yolo.model_path) if self.yolo.model_path else "")
        self.settings.setValue("session/tab", self.tabs.currentIndex())
        self.settings.setValue("review/local_scope", self.local_audit_scope.currentData())
        self.settings.setValue("review/profile", self.audit_profile.currentData())
        self.settings.sync()

    def closeEvent(self, event):
        if self.current and self.current.dirty: self.save_current()
        self._save_session()
        event.accept()
