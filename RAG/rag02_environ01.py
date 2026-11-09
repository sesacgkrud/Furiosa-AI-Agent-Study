# [목적] API 키를 ChatOpenAI 인자 대신 환경 변수 OPENAI_API_KEY 로 넘기기 (Day19)
# [흐름] os.environ 에 키 등록 → ChatOpenAI 가 인자 없이도 환경 변수에서 키를 찾아 쓴다 → invoke
# ※ 아래 키는 예시 값, ''' ''' 문자열은 출력 결과

from langchain_openai import ChatOpenAI
import os
# 이 프로그램이 실행되는 동안만 유효한 환경 변수 (프로그램이 끝나면 사라진다)
os.environ["OPENAI_API_KEY"] = "sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"   # 실제 키 대신 예시 값 (본인 키는 .env 에 보관)

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    # openai_api_key 를 넘기지 않아도 OPENAI_API_KEY 환경 변수를 자동으로 읽는다
    # openai_api_key=openai_api_key,
)

response = llm.invoke('안녕하세요.')
print(response)
'''
content='안녕하세요! 무엇을 도와드릴까요?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 9, 'total_tokens': 23, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-5.6-terra', 'system_fingerprint': None, 'id': 'chatcmpl-ETdLoVY9ccc2nxBIKfCx4rts0SKfR', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a0efe9-f993-7260-93d1-96cf6eabb18c-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 9, 'output_tokens': 14, 'total_tokens': 23, 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
'''
# print(response.content)

