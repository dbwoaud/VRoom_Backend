# VRoom 백엔드 서버 (AI 면접관 두뇌 · 4.2.4 백엔드 및 통신)

졸업과제 **VRoom**의 백엔드 서버 레포지토리입니다.
사용자 답변을 **LLM으로 채점**하고, 그 점수에 따라 **면접관 페르소나(긍정/중립/부정)를 실시간 가변**시키며,
다음 질문 대사 + 비언어 메타데이터(`Expression_ID`, `Gesture_ID`)를 **단일 JSON 패킷**으로 만들어
Unity로 보내고, 대사를 **TTS로 합성**해 음성까지 전달하는 "면접의 두뇌 + 통신 허브"입니다.

---

## 1. 전체 구조 개요

이 서버는 음성 처리 서버(Node A=STT, Node B=TTS) 사이에 들어가 AI 판단과 Unity 제어를 담당합니다.

```
Unity (마이크) ──ws:8000 ──▶ Node A(STT)
                                │ 전사 텍스트 + features를 POST (/process)
                                ▼
                        VRoom 백엔드(8080)  ── 채점 · 페르소나 가변 · 제스처/표정 선택 · 메모리
                            │                                   │
              대사 POST(TTS)│                       행동패킷+음성 │ ws:8080 (/ws/control)
                            ▼                                   ▼
                       Node B(TTS) ──음성──▶ 백엔드 ──────────▶ Unity
```

- 면접관이 **먼저** 자기소개를 요청(첫 발화) → 사용자 답변 → 채점 → 다음 질문 …
- 시나리오: 자기소개 → 기술질문 → 꼬리1(분기) → 꼬리2 → 인성 → 마무리 → 종합 피드백

---

## 2. 사용된 기술

- **FastAPI (async)**: WebSocket(Unity 제어 채널) + HTTP POST(STT 전사 수신) 엔드포인트
- **OpenAI / Groq SDK**: LLM 채점 + 대사 생성. `LLM_PROVIDER`로 전환 (최종 OpenAI, 테스트 Groq)
- **WebSocket**: 행동 지시 패킷(JSON 텍스트) + 면접관 음성(바이너리 PCM)을 같은 채널로 전송
- **Pydantic**: 행동 패킷 / 피드백 스키마 검증

LLM은 단계/페르소나에 맞는 대사·점수·제스처를 **단일 JSON으로 강제 출력**합니다.
시나리오 진행과 페르소나 결정은 코드(`session.py`)가 통제하여 흐름이 흔들리지 않습니다.

---

## 3. 스크립트 설명 (`app/`)

- **main.py**: FastAPI 앱. WebSocket `/ws/control`, HTTP `POST /process`, `GET /health`
- **config.py**: `.env` 로딩 (LLM 키, TTS 주소, 동작 플래그)
- **domain.py**: 면접 단계(Stage) · 페르소나(Persona) · `Expression_ID`/`Gesture_ID` 코드표 · 스키마
- **llm.py**: LLM 구조화 출력(대사 + 채점 + 제스처) 및 피드백 생성
- **session.py**: 면접 상태머신 + 점수→페르소나 가변 + 대화 메모리 + 피드백 집계
- **tts_client.py**: 대사를 구(phrase) 단위로 Node B(TTS)에 보내 음성 스트림 수신

---

## 4. 실행 방법

### 4.1 준비물
1. Python 3.10+
2. LLM API 키 (OpenAI `sk-...` 또는 Groq `gsk_...`)
3. STT/TTS 워커 + NVIDIA GPU + 모델 파일

### 4.2 .env 파일 형식
`.env` 작성:
```ini
LLM_PROVIDER=groq                 # 테스트: groq / 최종: openai
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
OPENAI_API_KEY=sk-...             # 최종 통합 시
OPENAI_MODEL=gpt-4o-mini

TTS_WORKER_URL=http://host.docker.internal:8001/process   # Node B(TTS) 주소
SKIP_TTS=false                    # true면 음성 생략(로직만 테스트)
PROXY_AUDIO_TO_STT=true           # Node A가 음성을 되받는 기존 구조면 true 권장
HOST=0.0.0.0
PORT=8080
```

### 4.3 설치 및 실행
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```
확인: 브라우저에서 `http://127.0.0.1:8080/health` → `{"status":"ok",...}`,
`http://127.0.0.1:8080/docs`에서 전체 API 확인.

### 4.4 백엔드 단독 테스트 (음성 서버 없이)
`.env`에 `SKIP_TTS=true`로 두면 TTS 없이 채점·페르소나·다음 질문 로직만 검증됩니다.
WebSocket으로 `init`을 보낸 뒤(면접관 첫 질문 수신), 사용자 답변을 흉내 내 POST:
```bash
curl -X POST http://127.0.0.1:8080/process -H "Content-Type: application/json" ^
 -d "{\"text\":\"3년차 백엔드 개발자입니다\",\"features\":{\"speakingTime\":12,\"pauseCount\":1}}"
```

---

## 5. 음성 처리 서버(Node A/B)와 통합

면접의 두뇌는 백엔드 하나로 모읍니다. **Node B는 TTS 전용으로 전환**하세요.

1. **Node A** `.env`: `TTS_WORKER_URL`을 **백엔드(8080)** 로 변경
   ```ini
   TTS_WORKER_URL = http://host.docker.internal:8080/process
   ```
   (가능하면 POST 본문에 `features`도 함께: `{"text":..., "features":{"speakingTime":..,"pauseCount":..,"averageVolume":..}}`. 단일 사용자면 `session_id`는 생략 가능)

2. **Node B**: 자체 LLM 호출을 끄고 **받은 text를 그대로 TTS**. 되돌리기 쉽게 env 플래그 권장:
   ```python
   TTS_ONLY = os.getenv("TTS_ONLY","false").lower()=="true"
   spoken_text = req.text if TTS_ONLY else call_llm(req.text)   # TTS 부분은 그대로
   ```
   `tts-worker-docker/.env`에 `TTS_ONLY=true`.

3. 백엔드 `.env`: `PROXY_AUDIO_TO_STT=true`(Node A 음성 패스스루 구조 유지), `TTS_WORKER_URL`=진짜 Node B(8001).

> 되돌리기: Node B `TTS_ONLY=false`, Node A `TTS_WORKER_URL`을 8001로 → 기존 프로토타입 복귀.

---

## 6. LLM 서버와 통합

현재 LLM 호출은 `app/llm.py` 한 곳에 캡슐화돼 있습니다(OpenAI/Groq SDK 직접 호출).
별도 LLM 서버/프롬프트 체계로 교체할 지점은 다음 두 함수입니다.

- `generate_turn(...)`: 단계·페르소나·대화기록을 받아 **대사 + 0~100 점수 + 제스처/표정 ID**를 JSON으로 반환
- `generate_feedback(...)`: 전체 기록 → 강점/개선점/총평

이 두 함수의 **입출력 계약(반환 스키마 `LLMTurn`, 피드백 dict)만 유지**하면, 내부를 
프롬프트/모델/체인으로 바꿔도 나머지(상태머신·통신)는 그대로 동작합니다.
페르소나 결정(점수 임계값)·시나리오 진행은 `session.py`가 통제하므로, LLM은 "현재 턴 생성"에 집중하면 됩니다.

---

## 7. 통신 명세

### 엔드포인트
- `WS /ws/control` — Unity ↔ 백엔드 (행동 패킷 + 음성)
- `POST /process` — Node A → 백엔드 (전사 텍스트 전달)
- `GET /health` — 상태 확인

### Unity → 백엔드 (WS 텍스트)
- `{"type":"init","session_id":"default","company":"네이버","job_title":"백엔드","resume":"..."}`
- `{"type":"request_feedback","session_id":"default"}`

### Node A → 백엔드 (HTTP)
- `POST /process` body: `{"session_id":"default"(생략가능),"text":"...","features":{...}}`

### 백엔드 → Unity (WS)
- 텍스트(행동 패킷, JsonUtility 호환):
  ```json
  {"type":"interviewer_turn","session_id":"default","stage":"FOLLOWUP_1",
   "persona":"NEGATIVE","dialogue":"...","expression_id":2,"gesture_id":3,
   "score":42,"is_final":false}
  ```
- `{"type":"thinking",...}` — LLM 연산 중 '검토 중' 더미 모션
- 바이너리 프레임 = 면접관 음성(44.1kHz Mono 32-bit Float PCM)
- `{"type":"audio_end"}` — 한 발화 음성 종료
- `{"type":"feedback_report","overall_score":..,"stage_scores":[...],"strengths":"..","improvements":"..","summary":".."}`

### Expression_ID / Gesture_ID 코드표 (Unity Animator와 약속)
| Expression_ID | 표정 | Gesture_ID | 동작 |
|:-:|:--|:-:|:--|
| 0 | 무표정 | 0 | Idle |
| 1 | 온화한 미소(긍정) | 1 | 깊게 끄덕임(긍정) |
| 2 | 미간 찌푸림(부정) | 2 | 고개 갸우뚱 |
| 3 | 경청/관심 | 3 | 팔짱(강한 부정) |
| 4 | 생각 중 | 4 | 펜 만지작 |
|   |  | 5 | 시작 안내 |
|   |  | 6 | 이력서 검토(=생각 중) |
|   |  | 7 | 경청 끄덕임 |

페르소나 규칙: 점수 70↑ 긍정 / 40~69 중립 / 40 미만 부정, 저점 2회 연속 시 강한 부정 고착.

---

## 9. 파일 구조
```
VRoom_Backend/
├── app/{main,config,domain,llm,session,tts_client}.py
├── requirements.txt
└── .env
```
