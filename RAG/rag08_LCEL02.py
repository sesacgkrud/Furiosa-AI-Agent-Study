# LCEL = Langchain Expression Language
# Chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

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

input = {'topic' : '양자 컴퓨터 학습 원리', 'how' : '초등학생도 이해하기 쉽게'}
response = chain.invoke(input)
print(response.content)