"""
면접 세션 상태머신. 사용자별로 하나씩 만들어 들고 있으며,
- 현재 단계(Stage) 진행
- 점수 -> 페르소나(긍정/중립/부정) 가변 전환 + 연속 저점 시 압박 고착
- 대화 기록(메모리) 누적
- 멀티모달 피쳐(발화시간/침묵 등) 집계
- 종료 시 피드백 산출
을 담당한다.

메모리는 면접이 6턴 내외로 짧으므로 전체 기록을 그대로 보관한다.
(세션이 길어지면 여기서 요약 압축 = LangChain Summary Memory 역할을 넣으면 된다.)
"""
from __future__ import annotations

from . import llm
from .domain import (
    BehaviorPacket,
    FeedbackReport,
    LLMTurn,
    Persona,
    Stage,
    STAGE_ORDER,
    StageScore,
    persona_from_score,
)


class InterviewSession:
    def __init__(self, session_id: str, company: str = "", job_title: str = "", resume: str = ""):
        self.session_id = session_id
        self.company = company
        self.job_title = job_title
        self.resume = resume

        self.stage: Stage = Stage.INIT
        self.persona: Persona = Persona.NEUTRAL
        self.consecutive_low = 0  # 연속 저점 카운트 (압박 고착용)

        self.turns: list[dict] = []          # {"role","stage","text"}
        self.stage_scores: list[tuple[str, int]] = []
        self.speaking_times: list[float] = []
        self.total_pauses = 0

    # ----- 메모리 -----
    def _history_text(self) -> str:
        lines = []
        for t in self.turns:
            who = "면접관" if t["role"] == "interviewer" else "지원자"
            lines.append(f"{who}({t['stage']}): {t['text']}")
        return "\n".join(lines)

    def _record(self, role: str, text: str):
        self.turns.append({"role": role, "stage": self.stage.value, "text": text})

    def _advance_stage(self):
        """다음 단계로 한 칸 이동."""
        if self.stage == Stage.INIT:
            self.stage = STAGE_ORDER[0]
            return
        idx = STAGE_ORDER.index(self.stage)
        self.stage = STAGE_ORDER[min(idx + 1, len(STAGE_ORDER) - 1)]

    # ----- 핵심: 면접관의 다음 발화 생성 -----
    async def first_question(self) -> BehaviorPacket:
        """면접 시작: 면접관이 먼저 자기소개를 요청하는 첫 발화."""
        self._advance_stage()  # INIT -> SELF_INTRO
        turn = await llm.generate_turn(
            stage=self.stage, persona=self.persona,
            company=self.company, job_title=self.job_title, resume=self.resume,
            history="", user_answer="",
        )
        self._record("interviewer", turn.dialogue)
        return self._to_packet(turn, is_final=False)

    async def on_user_answer(self, text: str, features: dict) -> BehaviorPacket:
        """사용자 답변(STT 결과)을 받아 채점하고 다음 단계 발화를 생성."""
        self._record("user", text)
        self._collect_features(features)

        was_closing = self.stage == Stage.CLOSING
        self._advance_stage()
        if was_closing:
            # 마무리 답변까지 끝남 -> DONE. 짧은 종료 멘트만.
            self.stage = Stage.DONE
            closing = BehaviorPacket(
                session_id=self.session_id, stage=Stage.DONE.value,
                persona=self.persona.value,
                dialogue="면접에 응해 주셔서 감사합니다. 잠시 후 결과를 안내해 드리겠습니다.",
                expression_id=1, gesture_id=1, score=-1, is_final=True,
            )
            self._record("interviewer", closing.dialogue)
            return closing

        # 직전 답변을 채점 + 다음 질문 생성
        turn = await llm.generate_turn(
            stage=self.stage, persona=self.persona,
            company=self.company, job_title=self.job_title, resume=self.resume,
            history=self._history_text(), user_answer=text,
        )

        # 점수 -> 페르소나 가변 전환 (코드가 최종 결정)
        if turn.score >= 0:
            prev_stage_name = self.turns[-2]["stage"] if len(self.turns) >= 2 else self.stage.value
            self.stage_scores.append((prev_stage_name, turn.score))
            self.consecutive_low = self.consecutive_low + 1 if turn.score < 40 else 0
            self.persona = persona_from_score(turn.score, self.consecutive_low)
            # 압박 고착: 연속 2회 이상 저점이면 강한 부정 제스처 강제
            turn = llm._clamp_to_set(turn, self.persona)
            if self.persona == Persona.NEGATIVE and self.consecutive_low >= 2:
                from .domain import ExpressionID, GestureID
                turn.expression_id = ExpressionID.SLIGHT_FROWN.value
                turn.gesture_id = GestureID.ARMS_CROSSED.value

        self._record("interviewer", turn.dialogue)
        return self._to_packet(turn, is_final=False)

    def _to_packet(self, turn: LLMTurn, is_final: bool) -> BehaviorPacket:
        return BehaviorPacket(
            session_id=self.session_id,
            stage=self.stage.value,
            persona=self.persona.value,
            dialogue=turn.dialogue,
            expression_id=turn.expression_id,
            gesture_id=turn.gesture_id,
            score=turn.score,
            is_final=is_final,
        )

    # ----- 멀티모달 피쳐 집계 -----
    def _collect_features(self, features: dict):
        if not features:
            return
        st = features.get("speakingTime")
        if isinstance(st, (int, float)):
            self.speaking_times.append(float(st))
        pc = features.get("pauseCount")
        if isinstance(pc, (int, float)):
            self.total_pauses += int(pc)

    # ----- 종료 피드백 -----
    async def build_feedback(self) -> FeedbackReport:
        avg_speak = (sum(self.speaking_times) / len(self.speaking_times)) if self.speaking_times else 0.0
        data = await llm.generate_feedback(
            company=self.company, job_title=self.job_title,
            transcript=self._history_text(), stage_scores=self.stage_scores,
            avg_speaking_time=avg_speak, total_pauses=self.total_pauses,
        )
        return FeedbackReport(
            session_id=self.session_id,
            overall_score=data.get("overall_score", 0),
            stage_scores=[StageScore(stage=s, score=v) for s, v in self.stage_scores],
            strengths=data.get("strengths", ""),
            improvements=data.get("improvements", ""),
            summary=data.get("summary", ""),
            avg_speaking_time=round(avg_speak, 1),
            total_pauses=self.total_pauses,
        )
