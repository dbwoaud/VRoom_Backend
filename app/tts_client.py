"""
TTS 워커(Node B) 연동. 면접관 대사 텍스트를 Node B로 POST 하고,
돌아오는 44.1kHz / Mono / 32-bit Float PCM 오디오 스트림을 청크 단위로 흘려준다.
(음성 처리 서버 Readme의 출력 포맷과 동일)

저지연을 위해 대사를 구(phrase) 단위로 잘라 순차 합성한다 = 스트리밍 처리.
"""
from __future__ import annotations

import re
from typing import AsyncIterator

import httpx

from .config import settings

# 문장부호 기준으로 구를 나눠 TTS에 먼저 들어가는 구부터 빠르게 합성.
_SPLIT = re.compile(r"(?<=[.?!。…\n])\s+|(?<=[,，])\s+")


def split_phrases(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT.split(text) if p.strip()]
    return parts or [text]


async def synthesize_stream(dialogue: str) -> AsyncIterator[bytes]:
    """대사를 구 단위로 Node B에 보내고 PCM 청크를 순서대로 yield."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for phrase in split_phrases(dialogue):
            async with client.stream(
                "POST", settings.tts_worker_url, json={"text": phrase}
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
