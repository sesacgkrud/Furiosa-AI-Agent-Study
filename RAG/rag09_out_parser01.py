# LCEL = Langchain Expression Language
# Chain = prompt | model | output_parser

# parser : 분석하다, 답변을 다듬다

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip() # strip() 공백이나 줄바꿈을 무시 (오탈자로 인한 에러 방지용)
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = PromptTemplate.from_template('{topic}에 대해 쉽게 설명해주세요.')

model = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url, # 사제 키값을 사용할 때는 key와 URL을 함께 넣어줘야 함
)

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

chain = prompt | model | output_parser

input = {'topic', 'Langchain 원리'}
response = chain.invoke(input)
# print(response.content)      # output_parser 가 있으면 필요 없음. '.content'를 하면 에러 발생
print(response)