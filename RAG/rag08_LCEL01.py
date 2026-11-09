# [목적] LCEL 파이프(|) 로 prompt 와 model 을 이어 체인을 만들고 실행 (Day19)
# [흐름] PromptTemplate → ChatOpenAI → chain = prompt | model → chain.invoke(입력) → response.content
# ※ input 은 { 키 : 값 } 딕셔너리가 아니라 쉼표로 쓴 { 값, 값 } 집합(set) 이다
#   PromptTemplate 은 입력 변수가 1 개일 때 dict 가 아닌 값을 받으면 그 값을 통째로 변수에 넣어서 오류 없이 실행되지만,
#   프롬프트에는 "{'topic', '...'}" 처럼 집합 모양 문자열이 그대로 들어간다 (순서도 실행마다 바뀔 수 있다)
#   → 의도대로라면 {'키' : '값'} (rag08_LCEL02 처럼 콜론) 이어야 한다

# LCEL = Langchain Expression Language
# Chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# monorouter(사제 OpenAI 호환 서버) 키를 .env 에서 읽고 앞뒤 공백 · 줄바꿈 제거
api_key = os.environ['MONOROUTER_API_KEY'].strip() # strip() 공백이나 줄바꿈을 무시 (오탈자로 인한 에러 방지용)
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = PromptTemplate.from_template('{topic}에 대해 쉽게 설명해주세요.')

model = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url, # 사제 키값을 사용할 때는 key와 URL을 함께 넣어줘야 함
)

# prompt 의 결과(완성된 프롬프트)가 model 의 입력으로 자동 연결된다
chain = prompt | model

input = {'topic', '양자 컴퓨터 학습 원리'}
response = chain.invoke(input)
print(response.content)