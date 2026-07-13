from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path


def analyze_image(image_path: Path, question: str, model: str = "gpt-4.1-mini") -> str:
    """Optional visual advisor. Reads OPENAI_API_KEY only from the environment."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("未安装 openai Python 包") from exc
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("请先设置环境变量 OPENAI_API_KEY")
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = OpenAI().responses.create(
        model=model,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": question},
            {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"},
        ]}],
    )
    return response.output_text
