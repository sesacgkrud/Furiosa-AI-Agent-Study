# [목적] .env 파일에 적어 둔 키를 python-dotenv 의 load_dotenv() 로 불러와 ChatOpenAI 호출 (Day19)
# [흐름] load_dotenv() → .env 의 OPENAI_API_KEY 가 환경 변수로 등록 → ChatOpenAI 가 자동으로 읽음 → invoke
# ※ .env 는 .gitignore 에 있어 GitHub 에 올라가지 않는다 (키를 안전하게 보관하는 방법)

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
# 실행 위치(또는 상위 폴더)의 .env 파일을 찾아 키=값 을 환경 변수로 등록
load_dotenv()

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
)

# response = llm.invoke('안녕하세요.')
response = llm.invoke('openai api key 안전하게 보관하는 방법 알려줘')
# print(response)
# 아래 문자열은 rag02 의 '안녕하세요.' 출력 예시를 복사해 둔 것 (이번 질문의 결과가 아니다)
'''
content='안녕하세요! 무엇을 도와드릴까요?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 9, 'total_tokens': 23, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-5.6-terra', 'system_fingerprint': None, 'id': 'chatcmpl-ETdLoVY9ccc2nxBIKfCx4rts0SKfR', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a0efe9-f993-7260-93d1-96cf6eabb18c-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 9, 'output_tokens': 14, 'total_tokens': 23, 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
'''
print(response.content)

