# [목적] f01 의 OpenAI 클라이언트 코드를 LangChain ChatOpenAI 로 바꿔 furiosa-llm 서버(Qwen3-4B-FP8)를 호출 (Day45)
#        주석 처리된 부분이 f01 방식, 살아 있는 코드가 LangChain 방식 → 두 코드를 나란히 비교
# [흐름] ChatOpenAI(base_url=로컬 서버) → llm.invoke(질문) → response.content 출력
# 아래 문자열 : 실행 가능한 모델 메모 + 이번에 쓴 serve 명령

"""
### 현재 환경에서 실행되는 모델 ###
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8
furiosa-llm serve furiosa-ai/gpt-oss-20b
furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct (X)
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
furiosa-llm serve furiosa-ai/Qwen3.5-0.88 (X)
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8

furiosa-llm serve furiosa-ai/Qwen3-4B-FP8 \
    --devices npu:0 \
    --host 0.0.0.0 \
    --port 8000
"""

# from openai import OpenAI
from langchain_openai import ChatOpenAI

# client = OpenAI(                            # Request, Response 역할
#     base_url = "http://127.0.0.1:8000/v1",
#     api_key = "EMPTY"
# )

# # 서버에 로드된 모델 확인
# models = client.models.list()
# model_id = models.data[0].id

# LangChain 의 OpenAI 채팅 모델 래퍼 → base_url 만 바꾸면 RNGD 서버에 붙는다 (모델 이름은 생략)
llm = ChatOpenAI(                            # Request, Response 역할
    base_url = "http://127.0.0.1:8000/v1",
    api_key = "EMPTY"
)

# # 추론
# response = client.chat.completions.create(
#     model = model_id,
#     messages=[
#         {
#             "role" : "user",
#             "content" : "NPU와 GPU의 차이점을 설명해줘"
#         }
#     ],
#     max_tokens=3000
# )

# messages 리스트 대신 질문 문자열을 바로 넣는다 → AIMessage 객체가 돌아온다
response = llm.invoke("NPU와 GPU의 차이점을 설명해줘")

# print(response.choices[0].message.content)
# 답변 텍스트는 .content
print(response.content)