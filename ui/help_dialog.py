from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTabWidget, QTextBrowser, QVBoxLayout
from core.version import APP_VERSION


def _page(html: str) -> QTextBrowser:
    view = QTextBrowser()
    view.setStyleSheet("QTextBrowser{background:#151923;border:1px solid #354052;padding:12px}")
    view.setOpenExternalLinks(False)
    view.setHtml(
        "<style>body{font-family:'Microsoft YaHei';font-size:14px;color:#e7ebf2;line-height:1.65}"
        "h1{color:#67b7ff;font-size:24px}h2{color:#76d6a4;font-size:18px;margin-top:20px}"
        "h3{color:#f0c75e;font-size:15px}code{background:#111722;color:#8fd5ff;padding:2px 5px}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #46536a;padding:7px;text-align:left}"
        "th{background:#263349;color:#8fd5ff}.tip{background:#18283a;border-left:4px solid #3a8dde;padding:10px}"
        ".warn{background:#35291a;border-left:4px solid #d88a27;padding:10px}</style>" + html
    )
    return view


class HelpDialog(QDialog):
    """In-app QSflow handbook. It stays local and does not open a browser."""

    TAB_NAMES = ["快速开始", "标注与AI", "检查工作流", "快捷键", "常见问题", "关于"]

    def __init__(self, parent=None, initial_tab=0):
        super().__init__(parent)
        self.setWindowTitle("QSflow 帮助与使用手册")
        self.resize(960, 720)
        self.setMinimumSize(760, 560)
        tabs = QTabWidget()
        tabs.addTab(_page(self._quick_start()), self.TAB_NAMES[0])
        tabs.addTab(_page(self._annotation()), self.TAB_NAMES[1])
        tabs.addTab(_page(self._review()), self.TAB_NAMES[2])
        tabs.addTab(_page(self._shortcuts()), self.TAB_NAMES[3])
        tabs.addTab(_page(self._faq()), self.TAB_NAMES[4])
        tabs.addTab(_page(self._about()), self.TAB_NAMES[5])
        tabs.setCurrentIndex(max(0, min(initial_tab, tabs.count() - 1)))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
        layout = QVBoxLayout(self); layout.addWidget(tabs); layout.addWidget(buttons)
        self.setWindowModality(Qt.WindowModality.NonModal)

    @staticmethod
    def _quick_start():
        return """
        <h1>QSflow 快速开始</h1>
        <p>QSflow 是本地目标检测数据标注、AI自动标注、快速训练和标注质量检查工具。</p>
        <h2>1. 创建或打开项目</h2>
        <ol><li>选择 <b>文件 → 创建数据集项目</b>（<code>Ctrl+N</code>）。</li>
        <li>选择 Pascal VOC、YOLO Detection 或 COCO Detection。</li>
        <li>也可以用 <b>文件 → 打开数据集项目</b>（<code>Ctrl+O</code>）打开已有数据。</li></ol>
        <p class='tip'>VOC 项目的图片位于 <code>JPEGImages</code>，XML 位于 <code>Annotations</code>。QSflow 的内部状态保存在 <code>.avls</code>，不会改变标准数据格式。</p>
        <h2>2. 导入图片</h2>
        <p>使用“导入图片文件”导入单张/多张图片，或使用“导入图片文件夹（递归）”扫描文件夹及所有嵌套子文件夹。图片会复制进项目，重复内容自动跳过。</p>
        <h2>3. 最短标注流程</h2>
        <ol><li>在左侧选择图片。</li><li>在画布空白处按住左键拖动创建框。</li>
        <li>右侧选择类别；拖动框内部移动，拖动四角缩放。</li>
        <li>按 <code>Ctrl+S</code> 保存，按 <code>D</code> 下一张。</li></ol>
        <h2>4. 数据安全</h2>
        <p>“从项目移除图片”会把图片及标注移动到 <code>.avls/trash</code>，不会立即永久删除。API Key 只在本次运行使用，不写入项目。</p>
        """

    @staticmethod
    def _annotation():
        return """
        <h1>人工标注与 AI 助手</h1>
        <h2>人工标注</h2>
        <ul><li><b>创建：</b>画布空白处拖动鼠标。</li><li><b>选择：</b>单击框。</li>
        <li><b>移动：</b>拖动框内部。</li><li><b>缩放：</b>拖动四角控制点。</li>
        <li><b>删除：</b>选中框后按 <code>Delete</code>。</li></ul>
        <h2>类别</h2>
        <p><code>Q/E</code> 切换上/下一类别，<code>1–9</code> 直接选类别。不同类别自动使用不同颜色。删除类别时，软件会说明受影响的框数量。</p>
        <h2>自动标注</h2>
        <ol><li>在“AI助手”加载 <code>.pt/.onnx/.tflite/.engine</code> 等支持的检测模型。</li>
        <li>设置置信度。</li><li>自动标注当前图片，或批量标注未标注图片。</li>
        <li>虚线框表示 AI 预测尚未人工确认；人工确认后转为实线。</li></ol>
        <h2>快速训练</h2>
        <p>标注一部分图片后点击“用当前标注快速训练”。QSflow 会转换临时训练数据、优先使用 CUDA GPU，并保存新的 <code>best.pt</code>。后续训练可继续使用上一次模型作为基础权重。</p>
        <p class='warn'>自动标注只是候选结果。训练前应抽查类别、漏标、重复框和框范围。</p>
        """

    @staticmethod
    def _review():
        return """
        <h1>检查与质检工作流</h1>
        <h2>左侧第一层：工作队列</h2>
        <table><tr><th>队列</th><th>含义</th><th>应该使用的功能</th></tr>
        <tr><td>① 待检查</td><td>有标注，但尚未完成本轮检查</td><td>A 模型批量检查；B 人工查看当前图后按 V；C 已检查整批时批量人工通过</td></tr>
        <tr><td>② 待人工处理</td><td>模型发现疑似漏标、错类或框偏移</td><td>单击问题框后修改、删除，或按 R 忽略误报</td></tr>
        <tr><td>③ 待批量确认</td><td>问题已处理完，等待最终通过</td><td>点击③，只确认左侧当前可见图片</td></tr>
        <tr><td>✓ 检查完成</td><td>已由模型或人工完成检查</td><td>查看即可；修改标注后会重新进入①</td></tr></table>
        <h2>左侧第二层：标注状态</h2>
        <p>它只是进一步缩小范围，不代表下一步动作。若工作队列有数量但左侧为空，请把它改为“全部标注状态”。</p>
        <h2>处理问题框</h2>
        <ul><li><code>W</code>：采用模型推荐类别。</li><li><code>Q/E</code> 或 <code>1–9</code>：选择类别并确认当前问题框。</li>
        <li><code>Delete</code>：删除关联错误框，或驳回漏标建议。</li><li><code>R</code>：原标注正确，记住本次误报。</li></ul>
        <h2>减少误报</h2>
        <p>优先使用“保守（少误报）”。误报较多时使用“保守重筛”；QSflow 会提高置信度与类别判断门槛，并忽略低可靠、贴边或重复候选。</p>
        <p class='tip'>名称含“左侧”的批量操作严格只处理左侧当前可见列表；左侧为空时不会处理隐藏图片。</p>
        """

    @staticmethod
    def _shortcuts():
        return """
        <h1>快捷键速查</h1>
        <table><tr><th>快捷键</th><th>作用</th><th>使用条件</th></tr>
        <tr><td>Ctrl+N</td><td>创建项目</td><td>任意位置</td></tr>
        <tr><td>Ctrl+O</td><td>打开项目</td><td>任意位置</td></tr>
        <tr><td>Ctrl+S</td><td>保存当前标注</td><td>已打开项目</td></tr>
        <tr><td>A / D</td><td>上一张 / 下一张</td><td>图片列表有内容</td></tr>
        <tr><td>Delete</td><td>删除选中标注框；图片列表聚焦时移除图片</td><td>先单击画布框或图片列表</td></tr>
        <tr><td>Q / E</td><td>上一类别 / 下一类别</td><td>先选择标注框或问题框</td></tr>
        <tr><td>1–9</td><td>直接选择第 1–9 个类别</td><td>先选择框</td></tr>
        <tr><td>V</td><td>人工确认当前整张图片</td><td>画布聚焦；①队列会自动显示下一张</td></tr>
        <tr><td>Enter</td><td>把当前类别应用到问题框</td><td>先选择问题框</td></tr>
        <tr><td>W</td><td>采用问题框推荐类别</td><td>问题含推荐类别</td></tr>
        <tr><td>R</td><td>忽略质检误报并记住</td><td>先选择问题框</td></tr>
        <tr><td>X</td><td>开关十字虚线定位辅助</td><td>任意位置</td></tr>
        <tr><td>鼠标滚轮</td><td>缩放图片</td><td>鼠标位于画布</td></tr>
        <tr><td>Ctrl+左键拖动</td><td>平移图片</td><td>画布内</td></tr>
        <tr><td>F1</td><td>打开本帮助</td><td>任意位置</td></tr></table>
        <p class='tip'>快捷键没有反应时，先单击画布或目标框，让画布获得焦点。</p>
        """

    @staticmethod
    def _faq():
        return """
        <h1>常见问题</h1>
        <h3>左侧明明有队列数量，列表却为空？</h3><p>第二层“标注状态”可能把范围筛空。切换为“全部标注状态”。批量按钮不会处理左侧看不到的图片。</p>
        <h3>AI 检查提示很多并非问题的框？</h3><p>选择保守模式并执行保守重筛。选中误报框按 <code>R</code>，软件会记住并避免重复提示。</p>
        <h3>为什么 AI 框是虚线？</h3><p>虚线表示 AI 预测尚未人工确认；按 <code>V</code> 或完成检查后会变成已确认状态。</p>
        <h3>TFLite 报输入尺寸不匹配？</h3><p>不同 TFLite 模型可能要求 112×112、320×320 或 640×640。QSflow 会读取真实输入尺寸。OpenART 带 <code>with_post_processing</code> 的设备自定义算子版本不能在普通电脑 LiteRT 中运行，应选择不带后处理的模型。</p>
        <h3>训练日志显示 0G 是否用了 GPU？</h3><p>没有。正常 CUDA 日志应显示 <code>CUDA:0</code>、显卡型号和非零 GPU 显存。QSflow 会优先选择 CUDA，环境无 CUDA 时才回退 CPU。</p>
        <h3>ChatGPT Plus 能否直接当 API 额度？</h3><p>不能。ChatGPT/Codex 套餐与 OpenAI API 计费独立。API 的 429 insufficient_quota 表示 API 项目没有可用额度。也可使用兼容接口的其他视觉模型。</p>
        <h3>导入新图片后会重复检查旧图片吗？</h3><p>默认“仅检查未检查图片”不会重复检查旧图片；选择“重新检查全部”才会重新运行。</p>
        <h3>软件会记住什么？</h3><p>窗口布局、面板宽度、最近项目、当前图片、类别、工作队列、标注筛选、置信度与检查策略。每个项目单独记住自己的模型，用户配置保存在 Windows 当前用户目录，不会污染分享包。</p>
        <h3>其他电脑报错时如何排查？</h3><p>选择“帮助 → 打开诊断日志目录”，把 <code>qsflow.log</code> 发给维护者。日志采用轮转保存，不包含 API Key。</p>
        """

    @staticmethod
    def _about():
        return f"""
        <h1>QSflow {APP_VERSION}</h1><h2>K&amp;C · AI Vision Dataset Workflow</h2>
        <p>面向目标检测数据集的本地标注、自动标注、快速训练与人在回路质量检查桌面工具。</p>
        <p>支持 Pascal VOC、YOLO Detection、COCO Detection，以及 Ultralytics 可识别的多种推理模型格式。</p>
        <p>Copyright © 2026 K&amp;C and QSflow contributors.<br>
        本程序依据 GNU AGPL-3.0-or-later 发布，不提供任何形式的担保。完整源码与许可证：
        <a href='https://github.com/yulianQS/QSflow'>github.com/yulianQS/QSflow</a></p>
        <p class='tip'>核心原则：数据格式标准化、人工与模型路径可自由选择、所有批量操作范围清晰可见。</p>
        """
