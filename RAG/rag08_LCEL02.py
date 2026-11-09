# [목적] 입력 변수가 2 개({topic}, {how}) 인 프롬프트로 LCEL 체인 실행 (Day19)
# [흐름] PromptTemplate → chain = prompt | model → invoke({'topic' : ..., 'how' : ...}) → response.content

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

prompt = PromptTemplate.from_template('{topic}에 대해 {how} 설명해주세요.')

model = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url, # 사제 키값을 사용할 때는 key와 URL을 함께 넣어줘야 함
)

chain = prompt | model

# 변수 이름을 키로 하는 딕셔너리 → 두 자리에 각각 채워진다
input = {'topic' : '양자 컴퓨터 학습 원리', 'how' : '초등학생도 이해하기 쉽게'}
response = chain.invoke(input)
print(response.content)