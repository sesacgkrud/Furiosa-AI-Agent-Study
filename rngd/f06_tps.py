from langchain_openai import ChatOpenAI
import time

max_tokens = 1000

llm = ChatOpenAI(                            # Request, Response 역할
    base_url = "http://127.0.0.1:8000/v1",
    api_key = "EMPTY",
    max_tokens = max_tokens,
    temperature = 0,
    top_p = 0.9,
)

start = time.perf_counter()
response = llm.invoke("NPU와 GPU의 차이점을 설명해줘")
end = time.perf_counter()

elapsed = round(end - start, 2)

# print(response.content)
print("추론 시간 : ", elapsed, "초")
print("출력 토큰 : ", max_tokens)
print("평균 처리량 : ", round(max_tokens/elapsed, 2), "tokens/s")