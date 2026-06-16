"""
VRoom 백엔드 서버 (착수보고서 4.2.4 백엔드 서버 및 통신)

역할:
  - Unity 와 WebSocket 양방향 통신(/ws/control): 행동 지시 패킷(JSON 텍스트 프레임)
    과 면접관 음성(바이너리 PCM 프레임)을 같은 채널로 내려보낸다.
  - STT 워커(Node A)가 전사 텍스트를 POST(/process) 하면, LLM 두뇌(session/llm)를
    돌려 채점 + 다음 질문 + 비언어 메타데이터를 만든 뒤
        1) 행동 패킷을 Unity로 push        (제스처/표정 먼저 트리거)
        2) 대사를 TTS(Node B)로 합성해 음성을 Unity로 스트리밍
    하는 오케스트레이션을 수행한다.

  데이터 흐름:
    Unity --(mic audio WS)--> Node A(STT) --(POST text+features)--> [이 서버]
    Unity <--(control WS: JSON 패킷 + PCM 오디오)-- [이 서버] --(POST 대사)--> Node B(TTS)
"""
from __future__ import annotations

import asyncio
import json
import websockets

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from .config import settings
from .domain import AnswerRequest, BehaviorPacket, ExpressionID, GestureID
from .session import InterviewSession
from . import tts_client

app = FastAPI(title="VRoom Backend", version="1.0")


# ---------------------------------------------------------------------------
# 세션 / WebSocket 레지스트리
# ---------------------------------------------------------------------------
class Hub:
    def __init__(self):
        self.sessions: dict[str, InterviewSession] = {}
        self.sockets: dict[str, WebSocket] = {}
        self.last_active: str | None = None  # session_id 미지정 POST 라우팅용
        self.lock = asyncio.Lock()

    def register(self, sid: str, ws: WebSocket):
        self.sockets[sid] = ws
        self.last_active = sid

    def unregister(self, sid: str):
        self.sockets.pop(sid, None)

    async def send_packet(self, sid: str, packet: BehaviorPacket):
        ws = self.sockets.get(sid)
        if ws:
            try:
                await ws.send_text(packet.model_dump_json())
            except Exception:
                self.unregister(sid)

    async def send_json(self, sid: str, obj: dict):
        ws = self.sockets.get(sid)
        if ws:
            try:
                await ws.send_text(json.dumps(obj, ensure_ascii=False))
            except Exception:
                self.unregister(sid)

    async def send_audio(self, sid: str, chunk: bytes):
        ws = self.sockets.get(sid)
        if ws:
            try:
                await ws.send_bytes(chunk)
            except Exception:
                self.unregister(sid)


hub = Hub()


async def speak(sid: str, packet: BehaviorPacket):
    """행동 패킷 push -> TTS 합성 -> 음성 스트리밍 -> 종료 신호."""
    await hub.send_packet(sid, packet)                     # 1) 제스처/표정/대사 먼저
    
    if settings.skip_tts:                                  # TTS 생략 모드 (Node B 없이 테스트)
        await hub.send_json(sid, {"type": "audio_end"})
        return

    try:
        async for chunk in tts_client.synthesize_stream(packet.dialogue):  # 2) 음성
            await hub.send_audio(sid, chunk)
    except Exception as e:
        print(f"[TTS 연결 실패 - 음성 생략] {e}")           # 에러가 나도 서버는 안 죽음
    await hub.send_json(sid, {"type": "audio_end"})        # 3) 한 발화 끝     # 3) 한 발화 끝


# ---------------------------------------------------------------------------
# WebSocket: Unity 제어 채널
# ---------------------------------------------------------------------------
@app.websocket("/ws/control")
async def ws_control(ws: WebSocket):
    await ws.accept()
    sid: str | None = None
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "init":
                sid = msg.get("session_id") or "default"
                hub.sessions[sid] = InterviewSession(
                    session_id=sid,
                    company=msg.get("company", ""),
                    job_title=msg.get("job_title", ""),
                    resume=msg.get("resume", ""),
                )
                hub.register(sid, ws)
                # 면접관이 먼저 자기소개를 요청 (첫 발화 + 음성)
                packet = await hub.sessions[sid].first_question()
                await speak(sid, packet)

            elif mtype == "utterance_end":
                # (옵션) Unity가 STT를 거치지 않고 직접 피쳐만 보낼 때 집계.
                if sid and sid in hub.sessions:
                    hub.sessions[sid]._collect_features(msg.get("features", {}))

            elif mtype == "request_feedback":
                if sid and sid in hub.sessions:
                    report = await hub.sessions[sid].build_feedback()
                    await hub.send_json(sid, report.model_dump())

    except WebSocketDisconnect:
        pass
    finally:
        if sid:
            hub.unregister(sid)


# ---------------------------------------------------------------------------
# HTTP: STT 워커(Node A) -> 백엔드 전사 텍스트 전달
#   Node A 의 .env TTS_WORKER_URL 을 이 엔드포인트로 바꾸면 된다.
# ---------------------------------------------------------------------------
@app.post("/process")
async def process(req: AnswerRequest):
    sid = req.session_id or hub.last_active or "default"
    session = hub.sessions.get(sid)
    if session is None:
        return JSONResponse({"error": "no active session. Unity must send 'init' first."}, status_code=409)

    print(f"[STT→백엔드 수신] {req.text}")
    
    # '생각 중' 더미 모션을 즉시 띄워 인지적 대기시간을 가린다 (RTT 제어).
    await hub.send_packet(sid, BehaviorPacket(
        type="thinking", session_id=sid, stage=session.stage.value,
        dialogue="", expression_id=ExpressionID.THINKING.value,
        gesture_id=GestureID.REVIEW_RESUME.value, score=-1,
    ))

    packet = await session.on_user_answer(req.text, req.features)
    print(f"[백엔드→TTS 대사] {packet.dialogue}  (stage={packet.stage}, persona={packet.persona}, score={packet.score})")
    # 호환 모드: Node A 가 음성을 되받길 기대하면 HTTP 응답으로 스트리밍.
    if settings.proxy_audio_to_stt:
        await hub.send_packet(sid, packet)

        async def audio_gen():
            async for chunk in tts_client.synthesize_stream(packet.dialogue):
                yield chunk
        return StreamingResponse(audio_gen(), media_type="application/octet-stream")

    # 기본 모드: 행동 패킷 + 음성 모두 백엔드->Unity WS 로 직접 전송.
    await speak(sid, packet)

    # 면접 종료면 피드백 리포트까지 이어서 push.
    if packet.is_final:
        report = await session.build_feedback()
        await hub.send_json(sid, report.model_dump())

    return {"ok": True, "stage": packet.stage, "persona": packet.persona, "score": packet.score}


@app.get("/health")
async def health():
    return {"status": "ok", "provider": settings.llm_provider, "active_sessions": len(hub.sessions)}

@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    """
    STT 워커가 '사용자 답변 텍스트'를 보내는 입구.
    백엔드가 채점/페르소나/질문 생성 후,
      - 자막+행동패킷을 Unity(/ws/control)로 push
      - 면접관 대사를 진짜 TTS(/ws/tts)로 합성해 음성을 STT로 릴레이
    STT 입장에선 기존 TTS와 동일하게 (음성청크 + {"type":"end"}) 를 받는다.
    """
    await ws.accept()
    print("[/ws/tts] STT 워커 연결됨")
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            user_text = msg.get("text", "")
            if not user_text:
                continue

            sid = hub.last_active or "default"
            session = hub.sessions.get(sid)
            if session is None:
                print("[/ws/tts] 활성 세션 없음 (Unity init 먼저 필요)")
                await ws.send_json({"type": "end"})
                continue

            print(f"[STT→백엔드 수신] {user_text}")

            # '생각 중' 모션을 Unity로 먼저
            await hub.send_packet(sid, BehaviorPacket(
                type="thinking", session_id=sid, stage=session.stage.value,
                dialogue="", expression_id=ExpressionID.THINKING.value,
                gesture_id=GestureID.REVIEW_RESUME.value, score=-1,
            ))

            # 채점 + 다음 질문 생성
            packet = await session.on_user_answer(user_text, {})
            print(f"[백엔드→TTS 대사] {packet.dialogue} "
                  f"(stage={packet.stage}, persona={packet.persona}, score={packet.score})")

            # 자막 + 행동패킷을 Unity로 (자막=면접관 대사)
            await hub.send_packet(sid, packet)

            # 면접관 대사를 진짜 TTS로 합성 → 음성을 STT로 릴레이
            try:
                async with websockets.connect(settings.tts_ws_url, max_size=None) as tts_ws:
                    for phrase in tts_client.split_phrases(packet.dialogue):
                        await tts_ws.send(json.dumps({"text": phrase}))
                        async for m in tts_ws:
                            if isinstance(m, str):
                                if json.loads(m).get("type") == "end":
                                    break
                            else:
                                await ws.send_bytes(m)   # 음성을 STT로 릴레이
            except Exception as e:
                print(f"[/ws/tts] TTS 릴레이 실패: {e}")

            # 한 발화 끝 신호 (STT가 이걸 받고 Unity VAD 잠금 해제)
            await ws.send_json({"type": "end"})

            # 면접 종료면 피드백도 Unity로
            if packet.is_final:
                report = await session.build_feedback()
                await hub.send_json(sid, report.model_dump())

    except WebSocketDisconnect:
        print("[/ws/tts] STT 워커 연결 종료")
    except Exception as e:
        print(f"[/ws/tts] Error: {e}")