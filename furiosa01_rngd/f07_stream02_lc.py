# [목적] f07_stream01 의 OpenAI 스트리밍을 LangChain llm.stream 으로 바꾼 버전 (Day45)
#        주석 처리된 부분이 OpenAI 방식, 맨 아래 두 줄이 LangChain 방식
# [흐름] ChatOpenAI → for chunk in llm.stream(질문) → chunk.content 출력

# from openai import OpenAI
from langchain_openai import ChatOpenAI

# client = OpenAI(                            # Request, Response 역할
#     base_url = "http://127.0.0.1:8000/v1",
#     api_key = "EMPTY"
# )

# model = client.models.list().data[0].id
llm = ChatOpenAI(                            # Request, Response 역할
    base_url = "http://127.0.0.1:8000/v1",
    api_key = "EMPTY",
    max_tokens = 1000,
    temperature = 0,
    top_p = 0.9,
)

# Streaming
# streaming = client.chat.completions.create(
#     model = model,
#     messages=[
#         {
#             "role" : "user",
#             "content" : "NPU와 GPU의 차이점을 설명해줘"
#         }
#     ],
#     max_tokens=3000,
#     stream=True
# )

# for chunk in streaming:
#     if chunk.choices:
#         content = chunk.choices[0].delta.content
#         if content:
#             print(content, end="", flush=True)

# llm.stream : invoke 와 같은 입력, 결과를 조각(AIMessageChunk) 으로 하나씩 돌려준다
for chunk in llm.stream("NPU와 GPU의 차이점을 설명해줘"):
    # chunk.content = 이번에 새로 생성된 글자 (None 대신 빈 문자열이라 그대로 출력해도 된다)
    print(chunk.content, end="", flush=True)