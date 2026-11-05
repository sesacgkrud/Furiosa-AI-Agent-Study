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
    stream=True
)

for chunk in streaming:
    if chunk.choices:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)