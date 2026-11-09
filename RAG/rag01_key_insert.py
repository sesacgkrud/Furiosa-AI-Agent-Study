# [목적] LangChain ChatOpenAI 로 OpenAI 대화 모델을 처음 호출해 본다 — API 키를 코드에 직접 넣는 방법 (Day19)
# [흐름] ChatOpenAI(모델 이름, temperature, openai_api_key) → llm.invoke(질문) → AIMessage 출력
# ※ 실제 키는 코드에 쓰지 않는다 (아래 값은 예시) → 다음 파일부터 환경 변수 · .env 로 옮긴다
# ※ 10 행의 invoke 를 주석 처리해서 11 행 print(response) 시점에는 response 가 없다 → NameError 가 나므로
#   처음 실행할 때는 10 행 주석을 풀어야 한다 (그 아래 ''' ''' 문자열은 그때의 출력 결과)

from langchain_openai import ChatOpenAI

openai_api_key = "sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"   # 실제 키 대신 예시 값 (본인 키는 .env 에 보관)
# OpenAI 대화 모델을 LangChain 방식으로 감싼 객체
llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    # 0 : 같은 질문에 거의 같은 답
    temperature=0,
    openai_api_key=openai_api_key,
)

# response = llm.invoke('안녕하세요.')
print(response)
'''
content='안녕하세요! 무엇을 도와드릴까요?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 9, 'total_tokens': 23, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-5.6-terra', 'system_fingerprint': None, 'id': 'chatcmpl-ETdLoVY9ccc2nxBIKfCx4rts0SKfR', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a0efe9-f993-7260-93d1-96cf6eabb18c-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 9, 'output_tokens': 14, 'total_tokens': 23, 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
'''
# print(response.content)

# LLM 은 이전 대화를 기억하지 않는다 → 두 번째 invoke 에는 첫 번째 질문이 들어 있지 않아 이름을 모른다고 답한다
response = llm.invoke('안녕 나는 하경이야')
response = llm.invoke('내 이름이 뭐게')
print(response)
# AIMessage 의 .content = 답변 글자만
print(response.content)