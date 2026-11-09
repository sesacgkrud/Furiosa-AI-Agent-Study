# [목적] OpenAI 클라이언트로 답변을 스트리밍 받기 — 다 만들어질 때까지 기다리지 않고 생성되는 조각(chunk)을 바로 출력 (Day45)
# [흐름] chat.completions.create(..., stream=True) → for chunk 반복 → chunk.choices[0].delta.content 출력

from openai import OpenAI

client = OpenAI(                            # Request, Response 역할
    base_url = "http://127.0.0.1:8000/v1",
    api_key = "EMPTY"
)

# 서버에 로드된 모델 확인
model = client.models.list().data[0].id

# 추론
streaming = client.chat.completions.create(
    model = model,
    messages=[
        {
            "role" : "user",
            "content" : "NPU와 GPU의 차이점을 설명해줘"
        }
    ],
    max_tokens=3000,
    # stream=True → 응답이 한 덩어리가 아니라 chunk 들의 반복(iterator)으로 온다
    stream=True
)

# chunk 를 받는 대로 하나씩 처리
for chunk in streaming:
    # choices 가 빈 chunk(마지막 사용량 정보 등)는 건너뛴다
    if chunk.choices:
        # 스트리밍에서는 message 대신 delta(이번에 새로 생긴 부분)에 글자가 들어 있다
        content = chunk.choices[0].delta.content
        # 첫 chunk(역할 정보) 처럼 content 가 None 인 경우는 출력하지 않는다
        if content:
            # end='' : 줄바꿈 없이 이어 붙임, flush=True : 버퍼에 모으지 않고 즉시 화면에 출력
            print(content, end="", flush=True)