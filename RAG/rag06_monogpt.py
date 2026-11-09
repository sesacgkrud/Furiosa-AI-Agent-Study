# [목적] OpenAI 대신 수업용 사제 서버(monorouter) 키 · 주소로 ChatOpenAI 를 호출 (Day19)
# [흐름] .env 의 MONOROUTER_API_KEY 읽기 → ChatOpenAI(api_key, base_url) → invoke
# ※ OpenAI 호환 서버는 base_url 만 바꾸면 같은 ChatOpenAI 코드로 쓸 수 있다 (이후 RAG · LangGraph 파일의 기본 설정)

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip() # strip() 공백이나 줄바꿈을 무시 (오탈자로 인한 에러 방지용)
# 아래 줄은 base_url 없이 사제 키를 넣었을 때 OpenAI 서버가 돌려준 오류(invalid_api_key) 메시지
# You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'
# 요청을 보낼 서버 주소 (OpenAI 호환)
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url, # 사제 키값을 사용할 때는 key와 URL을 함께 넣어줘야 함
)

# response = llm.invoke('안녕하세요.')
response = llm.invoke('안녕하세요')
# print(response)
'''
content='안녕하세요! 무엇을 도와드릴까요?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 9, 'total_tokens': 23, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-5.6-terra', 'system_fingerprint': None, 'id': 'chatcmpl-ETdLoVY9ccc2nxBIKfCx4rts0SKfR', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a0efe9-f993-7260-93d1-96cf6eabb18c-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 9, 'output_tokens': 14, 'total_tokens': 23, 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
'''
print(response.content)