"""AI daily digest generator using DeepSeek API."""

import json
from datetime import datetime, timezone

import requests

from ..collectors.base import CollectedItem


AI_SYSTEM_PROMPT = """你是 CunRadar 的 AI 日报编辑。你的任务是根据今天采集到的各项更新，生成一份简洁、有条理的中文日报。

输入内容可能包含英文（来自 GitHub、YouTube 等），你必须用中文总结所有内容。

规则：
1. 全文必须使用中文撰写，将英文内容翻译/总结为地道中文。
2. 按主题分类整理相关内容（如：大模型、AI 工具、开源项目、行业动态等）。
3. 每条内容用 1-3 句话说明：更新了什么，为什么值得关注。
4. GitHub 项目需注明仓库名和主要功能。
5. 视频内容需注明创作者和主题。
6. 使用 Markdown 格式，二级标题 (##) 分类，三级标题 (###) 细项。
7. 内容具体有料，避免空话套话。
8. 总字数 400-1000 字，保证信息密度。
9. 如果今天内容少（5条以内），仍写一份简短日报。"""


def build_user_prompt(items: list[CollectedItem], date_str: str) -> str:
    """Build the user prompt from collected items."""
    lines = [f"今日日期: {date_str}", "", "今天采集到的更新:"]
    for i, item in enumerate(items, 1):
        source_names = {"youtube": "YouTube", "bilibili": "B站", "rss": "RSS", "github": "GitHub", "github_trending": "GitHub热榜"}
        source_tag = source_names.get(item.source, item.source.upper())
        published = item.published.strftime("%H:%M UTC") if item.published else "未知时间"
        lines.append(f"{i}. [{source_tag}] {item.title}")
        lines.append(f"   来源: {item.source_name} | {published}")
        if item.description:
            desc = item.description[:200].replace("\n", " ")
            lines.append(f"   简介: {desc}")
        lines.append("")
    return "\n".join(lines)


def generate_digest(
    items: list[CollectedItem],
    date_str: str,
    api_key: str,
    model: str = "deepseek-chat",
    api_base: str = "https://api.deepseek.com",
    timeout: int = 120,
) -> str:
    """Generate an AI-powered daily digest using DeepSeek.

    Returns:
        The markdown digest text. Returns empty string on failure.
    """
    if not items:
        return "今日无新内容。"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(items, date_str)},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            f"{api_base.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        print(f"  [AI Digest] Generated successfully ({len(content)} chars)")
        return content
    except Exception as e:
        print(f"  [AI Digest] Failed: {e}")
        return ""
