from a_kingkeras import get_llm

llm = get_llm()

for chunk in llm.stream("NPU와 GPU의 차이점을 설명해줘"):
    print(chunk.content, end="", flush=True)