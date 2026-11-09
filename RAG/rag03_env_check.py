# [목적] 코드에서 키를 등록하지 않고, 윈도우 시스템 환경 변수에 저장해 둔 OPENAI_API_KEY 로 호출되는지 확인 (Day19)
# [흐름] (코드 밖에서 OPENAI_API_KEY 등록) → ChatOpenAI → invoke
# ※ 3 행 주석처럼 코드에 키를 쓰지 않아도 실행되면 시스템 환경 변수에서 읽은 것

from langchain_openai import ChatOpenAI
import os
# os.environ["OPENAI_API_KEY"] = "sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"   # 실제 키 대신 예시 값 (본인 키는 .env 에 보관)

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    # openai_api_key=openai_api_key,
)

# 키를 찾지 못하면 여기서 인증 오류가 난다
response = llm.invoke('안녕하세요.')
print(response)
'''
content='안녕하세요! 무엇을 도와드릴까요?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 9, 'total_tokens': 23, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-5.6-terra', 'system_fingerprint': None, 'id': 'chatcmpl-ETdLoVY9ccc2nxBIKfCx4rts0SKfR', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a0efe9-f993-7260-93d1-96cf6eabb18c-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 9, 'output_tokens': 14, 'total_tokens': 23, 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
'''
# print(response.content)

