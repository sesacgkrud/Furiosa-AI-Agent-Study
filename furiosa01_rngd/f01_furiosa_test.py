# [목적] furiosa-llm serve 로 RNGD(NPU) 에 띄운 모델에 OpenAI 클라이언트로 질문을 보내 답을 받는 첫 테스트 (Day42)
# [흐름] 1. OpenAI 클라이언트의 base_url 을 로컬 furiosa-llm 서버로 지정
#        2. client.models.list() 로 서버에 올라간 모델 이름 확인
#        3. chat.completions.create 로 질문 → response.choices[0].message.content 출력
# 아래 문자열은 이 환경에서 serve 해 본 모델 목록 메모 ((X) = 실행 안 됨)

"""
### 현재 환경에서 실행되는 모델 ###
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8
furiosa-llm serve furiosa-ai/gpt-oss-20b
furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct (X)
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
furiosa-llm serve furiosa-ai/Qwen3.5-0.88 (X)
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8
"""

# OpenAI 공식 파이썬 클라이언트 → OpenAI 호환 서버(furiosa-llm)에도 그대로 쓸 수 있다
from openai import OpenAI

client = OpenAI(                            # Request, Response 역할
    # OpenAI 서버 대신 RNGD 에 띄운 로컬 서버로 요청을 보낸다
    base_url = "http://127.0.0.1:8000/v1",
    # 로컬 서버는 키를 검사하지 않으므로 아무 값
    api_key = "EMPTY"
)

# 서버에 로드된 모델 확인
models = client.models.list()
# 서버에 모델이 하나만 떠 있으므로 첫 번째 모델의 id(이름) 를 쓴다
model_id = models.data[0].id

print("모델 :", model_id)

# 추론
# Chat Completions API : 대화 메시지 리스트를 보내 답변 생성
response = client.chat.completions.create(
    model = model_id,
    messages=[
        # role = user (사용자 질문), content = 질문 내용
        {
            "role" : "user",
            "content" : "NPU와 GPU의 차이점을 설명해줘"
        }
    ],
    # 답변 최대 토큰 수
    max_tokens=3000
)

# 답변 텍스트는 choices[0].message.content 에 들어 있다
print(response.choices[0].message.content)