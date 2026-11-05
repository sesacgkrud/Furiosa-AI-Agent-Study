from a_kingkeras import get_llm

llm = get_llm(
    max_tokens = 1000,
    temperature = 0.9,
    top_p = 1.0,
)

response = llm.invoke("NPU와 GPU의 차이점을 설명해줘")
print(response.content)