# [목적] get_llm 으로 만든 llm 의 답변을 스트리밍으로 출력 (Day47)
# [흐름] llm.stream(prompt) → chunk 가 올 때마다 chunk.content 를 줄바꿈 없이 즉시 출력

from kingkeras import get_llm

llm = get_llm(max_tokens=1000)

prompt = "인공지능의 발전 과정을 설명해줘."

# 답변을 조각(chunk) 단위로 받는다 → 첫 글자부터 바로 화면에 나온다
for chunk in llm.stream(prompt):
    # end='' : 줄바꿈 없이 이어 붙임, flush=True : 즉시 출력
    print(chunk.content, end="", flush=True)
