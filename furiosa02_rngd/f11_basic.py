# [목적] kingkeras 모듈의 get_llm 으로 Qwen3-8B-FP8 서버에 질문 하나를 보내는 기본 호출 (Day47)
# [흐름] 서버 실행(아래 문자열의 serve 명령) → get_llm(max_tokens=1000) → llm.invoke(prompt) → response.content
# 아래 문자열 : 이 실습에서 쓴 서버 실행 명령 (npu:0, 포트 8000)

"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
--devices npu:0 \
--host 0.0.0.0 \
--port 8000 \
"""


# 같은 폴더의 kingkeras.py 에서 get_llm 함수만 가져온다
from kingkeras import get_llm

# 답변 최대 1000 토큰, 나머지 옵션은 서버 기본값
llm = get_llm(max_tokens=1000)

prompt = "NPU와 GPU의 차이를 설명해줘."
# 질문 → 답변이 다 만들어질 때까지 기다렸다가 한 번에 받는다
response = llm.invoke(prompt)

# AIMessage 의 .content = 답변 텍스트
print(response.content)