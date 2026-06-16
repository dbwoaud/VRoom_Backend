"""
LLM 엔진. OpenAI/Groq 호환 SDK를 사용해 면접관 대사 + 채점 + 비언어 메타데이터를
단일 JSON으로 강제 출력(Structured Output)받는다.

핵심: 시나리오 진행(단계 전환)과 페르소나 결정은 코드(session.py)가 통제하고,
LLM은 '주어진 단계/페르소나에 맞는 대사·점수·제스처'만 생성한다.
이렇게 분리하면 그래프(면접 흐름)가 흔들리지 않아 디버깅이 쉽다.
"""
from __future__ import annotations

import json
from typing import List

from openai import AsyncOpenAI

from .config import settings
from .domain import (
    ExpressionID,
    GestureID,
    LLMTurn,
    Persona,
    PERSONA_BEHAVIOR_SET,
    Stage,
)

_base_url, _api_key, _model = settings.resolve_llm()
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """클라이언트를 첫 사용 시점에 생성(지연 초기화). 키가 없어도 임포트는 성공한다."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=_api_key or "missing-key", base_url=_base_url)
    return _client


# 단계별로 LLM에게 줄 '이번 턴에 해야 할 일' 지시문.
STAGE_TASK = {
    Stage.SELF_INTRO: "면접 시작 인사를 건네고, 타겟 기업/직무를 포함한 자기소개를 요청하라. 아직 채점 대상이 아니므로 score=-1.",
    Stage.TECH_Q1: "지원 직무의 핵심 역량을 검증하는 기술 질문 1개를 하라. 직전 답변(자기소개)은 채점하지 말고 score=-1.",
    Stage.FOLLOWUP_1: "직전 기술 답변의 정확도/논리성을 0~100으로 채점하라. 그 점수에 맞춰 심화(정확) 또는 반박/압박(부정확) 꼬리질문을 하라.",
    Stage.FOLLOWUP_2: "직전 답변을 STAR(행동/결과) 관점에서 재채점하고, 그에 맞는 2차 꼬리질문을 하라.",
    Stage.BEHAVIORAL: "직전 답변을 채점한 뒤, 협업/갈등해결 등 인성·조직적합성 질문으로 분위기를 자연스럽게 전환하라.",
    Stage.CLOSING: "직전 답변을 채점한 뒤, '마지막으로 하고 싶은 말이나 궁금한 점'을 묻는 마무리 질문을 하고 정중히 마친다.",
}


def _system_prompt(company: str, job_title: str, resume: str) -> str:
    return (
        "당신은 VR 모의면접의 전문 면접관이다. 다음 지원자를 면접한다.\n"
        f"- 지원 기업: {company or '미지정'}\n"
        f"- 지원 직무: {job_title or '미지정'}\n"
        f"- 이력서 요약: {resume or '미지정'}\n"
        "규칙:\n"
        "1) 모든 대사는 자연스러운 한국어 존댓말. 한 번에 한 가지 질문만.\n"
        "2) 대사는 음성합성(TTS)으로 출력되므로 마크다운/이모지/괄호설명 없이 말로만 작성.\n"
        "3) 반드시 지정된 JSON 스키마로만 응답한다. 그 외 텍스트 금지.\n"
    )


def _turn_instruction(stage: Stage, persona: Persona, history: str, user_answer: str) -> str:
    sets = PERSONA_BEHAVIOR_SET[persona]
    allowed_expr = ", ".join(f"{e.value}={e.name}" for e in sets["expressions"])
    allowed_gest = ", ".join(f"{g.value}={g.name}" for g in sets["gestures"])
    return (
        f"[현재 단계] {stage.value}\n"
        f"[이번 턴 과제] {STAGE_TASK[stage]}\n"
        f"[현재 페르소나] {persona.value} "
        "(POSITIVE=안정감/미소, NEUTRAL=경청, NEGATIVE=압박/냉정)\n"
        f"[직전 지원자 답변] {user_answer or '(아직 없음 - 면접관이 먼저 말함)'}\n"
        f"[지금까지의 대화 요약]\n{history or '(없음)'}\n\n"
        "아래 JSON 스키마로만 응답하라:\n"
        "{\n"
        '  "dialogue": "면접관 대사(한국어)",\n'
        '  "score": 정수(-1 또는 0~100; 채점 대상 아니면 -1),\n'
        '  "score_reason": "채점 근거 한 문장",\n'
        f'  "expression_id": 정수(허용값: {allowed_expr}),\n'
        f'  "gesture_id": 정수(허용값: {allowed_gest})\n'
        "}\n"
        "expression_id/gesture_id는 반드시 위 허용값 중에서만 고른다."
    )


async def _ask_json(system: str, user: str) -> dict:
    resp = await _get_client().chat.completions.create(
        model=_model,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def _clamp_to_set(turn: LLMTurn, persona: Persona) -> LLMTurn:
    """LLM이 허용 세트를 벗어난 ID를 내놓아도 안전하게 보정."""
    sets = PERSONA_BEHAVIOR_SET[persona]
    expr_ids = [e.value for e in sets["expressions"]]
    gest_ids = [g.value for g in sets["gestures"]]
    if turn.expression_id not in expr_ids:
        turn.expression_id = expr_ids[0]
    if turn.gesture_id not in gest_ids:
        turn.gesture_id = gest_ids[0]
    return turn


async def generate_turn(
    *, stage: Stage, persona: Persona, company: str, job_title: str, resume: str,
    history: str, user_answer: str,
) -> LLMTurn:
    """한 턴의 면접관 발화를 생성한다. JSON 파싱 실패 시 1회 재시도."""
    system = _system_prompt(company, job_title, resume)
    user = _turn_instruction(stage, persona, history, user_answer)
    for attempt in range(2):
        try:
            data = await _ask_json(system, user)
            turn = LLMTurn(**data)
            return _clamp_to_set(turn, persona)
        except Exception:
            if attempt == 1:
                # 최후의 안전망: 빈 응답 대신 기본 대사
                return LLMTurn(
                    dialogue="네, 잘 들었습니다. 다음 질문으로 넘어가겠습니다.",
                    score=-1,
                    expression_id=ExpressionID.NEUTRAL.value,
                    gesture_id=GestureID.LISTENING_NOD.value,
                )
    raise RuntimeError("unreachable")


async def generate_feedback(*, company: str, job_title: str, transcript: str,
                            stage_scores: list[tuple[str, int]],
                            avg_speaking_time: float, total_pauses: int) -> dict:
    """면접 전체를 종합한 피드백(강점/개선점/총평)을 JSON으로 생성."""
    scored = [s for _, s in stage_scores if s >= 0]
    overall = round(sum(scored) / len(scored)) if scored else 0
    system = "당신은 면접 코치다. 아래 면접 기록을 바탕으로 건설적 피드백을 한국어로 작성한다. JSON으로만 응답한다."
    user = (
        f"기업/직무: {company} / {job_title}\n"
        f"평균 발화시간(초): {avg_speaking_time:.1f}, 총 침묵 횟수: {total_pauses}\n"
        f"단계별 점수: {stage_scores}\n"
        f"전체 대화 기록:\n{transcript}\n\n"
        "다음 JSON으로만 응답:\n"
        '{ "strengths": "강점 2~3문장", "improvements": "개선점 2~3문장", "summary": "총평 2문장" }'
    )
    try:
        data = await _ask_json(system, user)
    except Exception:
        data = {"strengths": "", "improvements": "", "summary": ""}
    data["overall_score"] = overall
    return data
