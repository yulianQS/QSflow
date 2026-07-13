from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw

from core.models import ImageItem


@dataclass
class VisionProviderConfig:
    name: str = "OpenAI"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    protocol: str = "responses"  # responses | chat_completions
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str = ""  # session only; never persisted by the UI


AUDIT_SCHEMA = {
    "ok": True,
    "summary": "",
    "issues": [{
        "type": "missing_box|wrong_class|loose_box|tight_box|duplicate_box|invalid_box|other",
        "severity": "high|medium|low",
        "box_index": 1,
        "message": "",
        "suggested_label": None,
        "suggested_bbox": None,
    }],
}


def _annotated_data_url(item: ImageItem) -> str:
    buffer = BytesIO()
    _draw_annotated_image(item).save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _draw_annotated_image(item: ImageItem) -> Image.Image:
    image = Image.open(item.path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ("#ff3b5c", "#00d98b", "#3b9cff", "#ffb02e", "#d95cff", "#00ced1")
    for index, box in enumerate(item.boxes, 1):
        color = colors[(index - 1) % len(colors)]
        width = max(2, round(max(image.size) / 400))
        draw.rectangle((box.xmin, box.ymin, box.xmax, box.ymax), outline=color, width=width)
        text = f"#{index} {box.label}"
        text_box = draw.textbbox((box.xmin, box.ymin), text)
        draw.rectangle((text_box[0] - 2, text_box[1] - 2, text_box[2] + 3, text_box[3] + 3), fill=color)
        draw.text((box.xmin, box.ymin), text, fill="black")
    return image


def export_issue_image(project_root: Path, item: ImageItem, result: dict) -> Path | None:
    issues = result.get("issues", [])
    if not issues:
        return None
    image = _draw_annotated_image(item)
    draw = ImageDraw.Draw(image)
    width = max(3, round(max(image.size) / 260))
    for issue_index, issue in enumerate(issues, 1):
        bbox = issue.get("suggested_bbox")
        box_index = issue.get("box_index")
        if not bbox and isinstance(box_index, int) and 1 <= box_index <= len(item.boxes):
            box = item.boxes[box_index - 1]
            bbox = [box.xmin, box.ymin, box.xmax, box.ymax]
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [round(float(value)) for value in bbox]
        draw.rectangle((x1, y1, x2, y2), outline="#ff2020", width=width)
        text = f"ISSUE {issue_index}: {issue.get('type', 'other')}"
        text_box = draw.textbbox((x1, max(0, y1 - 16)), text)
        draw.rectangle((text_box[0] - 2, text_box[1] - 2, text_box[2] + 3, text_box[3] + 3), fill="#ff2020")
        draw.text((x1, max(0, y1 - 16)), text, fill="white")
    output_dir = project_root / ".avls" / "review_issues" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{item.path.stem}_issues.jpg"
    image.save(output, format="JPEG", quality=94)
    return output


def _parse_json(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start:end + 1]
    result = json.loads(cleaned)
    if not isinstance(result, dict) or not isinstance(result.get("issues", []), list):
        raise RuntimeError("模型返回的审查结果格式不正确")
    result.setdefault("ok", not result.get("issues"))
    result.setdefault("summary", "")
    return result


class VisionAuditService:
    def audit(self, item: ImageItem, classes: list[str], config: VisionProviderConfig) -> dict:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("未安装 openai Python 包") from exc
        key = config.api_key or os.getenv(config.api_key_env, "")
        if not key:
            raise RuntimeError(f"请输入 API Key，或设置环境变量 {config.api_key_env}")
        client = OpenAI(api_key=key, base_url=config.base_url.rstrip("/"), timeout=90)
        boxes = [{"index": i, "label": b.label, "bbox": [b.xmin, b.ymin, b.xmax, b.ymax]} for i, b in enumerate(item.boxes, 1)]
        prompt = (
            "你是目标检测数据标注质检员。检查图中带编号的矩形框与下面的标注数据。"
            "只检查可见事实，重点寻找漏标、错类、框过松、框过紧、重复框和无效框。"
            "允许类别只有：" + json.dumps(classes, ensure_ascii=False) + "。"
            f"图片尺寸：{item.width}x{item.height}。当前框：" + json.dumps(boxes, ensure_ascii=False) + "。"
            "返回一个 JSON 对象，不要 Markdown。格式示例：" + json.dumps(AUDIT_SCHEMA, ensure_ascii=False) + "。"
            "box_index 对漏标可为 null；suggested_bbox 使用 [xmin,ymin,xmax,ymax] 或 null。"
        )
        image_url = _annotated_data_url(item)
        if config.protocol == "responses":
            response = client.responses.create(model=config.model, input=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_url},
            ]}])
            text = response.output_text
        else:
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]}],
            )
            text = response.choices[0].message.content or ""
        return _parse_json(text)

    def audit_with_codex(self, item: ImageItem, classes: list[str]) -> dict:
        executable = self.find_working_codex()
        if not executable:
            raise RuntimeError("未找到可执行的独立 Codex CLI。WindowsApps 内的桌面版 EXE 不能被外部程序调用；请安装 Codex CLI 并使用 ChatGPT 登录。")
        boxes = [{"index": i, "label": b.label, "bbox": [b.xmin, b.ymin, b.xmax, b.ymax]} for i, b in enumerate(item.boxes, 1)]
        prompt = (
            "只做目标检测标注质量检查，不修改任何文件。检查附加图片中带编号的框，寻找漏标、错类、框过松、框过紧、重复框和无效框。"
            f"允许类别：{json.dumps(classes, ensure_ascii=False)}；图片尺寸：{item.width}x{item.height}；"
            f"当前框：{json.dumps(boxes, ensure_ascii=False)}。按指定 JSON Schema 返回，box_index 对漏标为 null。"
        )
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["ok", "summary", "issues"],
            "properties": {
                "ok": {"type": "boolean"}, "summary": {"type": "string"},
                "issues": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                    "required": ["type", "severity", "box_index", "message", "suggested_label", "suggested_bbox"],
                    "properties": {
                        "type": {"type": "string"}, "severity": {"type": "string"},
                        "box_index": {"type": ["integer", "null"]}, "message": {"type": "string"},
                        "suggested_label": {"type": ["string", "null"]},
                        "suggested_bbox": {"anyOf": [{"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}, {"type": "null"}]},
                    }}},
            },
        }
        with tempfile.TemporaryDirectory(prefix="avls_codex_") as temp:
            root = Path(temp)
            image_path, schema_path, output_path = root / "audit.jpg", root / "schema.json", root / "result.json"
            _draw_annotated_image(item).save(image_path, format="JPEG", quality=92)
            schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
            command = [executable, "exec", "--image", str(image_path), "--output-schema", str(schema_path),
                "--output-last-message", str(output_path), "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral", "-"]
            try:
                completed = subprocess.run(command, input=prompt, text=True, encoding="utf-8", errors="replace",
                    cwd=str(item.path.parent), capture_output=True, timeout=180, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except PermissionError as exc:
                raise RuntimeError("找到 Codex CLI，但 Windows 拒绝执行。请安装独立 Codex CLI，或检查 codex.exe 权限。") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("本机 Codex 质检超过 180 秒，已停止等待") from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()[-1200:]
                raise RuntimeError(f"Codex CLI 运行失败（代码 {completed.returncode}）：\n{detail}")
            if not output_path.exists():
                raise RuntimeError("Codex CLI 未生成质检结果")
            return _parse_json(output_path.read_text(encoding="utf-8-sig"))

    @staticmethod
    def find_working_codex() -> str | None:
        candidates: list[str] = []
        if os.name == "nt" and os.getenv("LOCALAPPDATA"):
            candidates.append(str(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe"))
        found = shutil.which("codex")
        if found:
            candidates.append(found)
        if os.name == "nt":
            try:
                result = subprocess.run(["where.exe", "codex"], capture_output=True, text=True, timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                candidates.extend(line.strip() for line in result.stdout.splitlines() if line.strip())
            except Exception:
                pass
        for candidate in dict.fromkeys(candidates):
            try:
                result = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=10,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if result.returncode == 0:
                    return candidate
            except (OSError, subprocess.SubprocessError):
                continue
        return None
