"""
### 현재 환경에서 실행되는 모델 ###
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8
furiosa-llm serve furiosa-ai/gpt-oss-20b
furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct (X)
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
furiosa-llm serve furiosa-ai/Qwen3.5-0.88 (X)
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8
"""

from openai import OpenAI

client = OpenAI(                            # Request, Response 역할
    base_url = "http://127.0.0.1:8000/v1",
    api_key = "EMPTY"
)

# 서버에 로드된 모델 확인
models = client.models.list()
model_id = models.data[0].id

print("모델 :", model_id)

# 추론
response = client.chat.completions.create(
    model = model_id,
    messages=[
        {
            "role" : "user",
            "content" : "NPU와 GPU의 차이점을 설명해줘"
        }
    ],
    max_tokens=3000
)

print(response.choices[0].message.content)