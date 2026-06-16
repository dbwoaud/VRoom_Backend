import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8080/ws/control") as ws:
        # 1) 면접 시작
        await ws.send(json.dumps({
            "type": "init", "session_id": "test1",
            "company": "네이버", "job_title": "백엔드 개발자",
            "resume": "Spring/Java 3년"
        }))
        # 2) 면접관 첫 질문(자기소개 요청)이 패킷으로 도착
        async def listen():
            async for raw in ws:
                if isinstance(raw, bytes):
                    print(f"[음성] {len(raw)} bytes")          # TTS PCM
                else:
                    print("[패킷]", raw)                        # 행동 패킷 / 피드백
        listen_task = asyncio.create_task(listen())
        await asyncio.sleep(8)
        listen_task.cancel()

asyncio.run(main())