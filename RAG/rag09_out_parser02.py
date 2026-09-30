# LCEL = Langchain Expression Language
# Chain = prompt | model | output_parser

# parser : 분석하다, 답변을 다듬다

template = '''
당신의 영어를 가르치는 10년차 영어 선생님입니다.
주어진 상황에 맞는 영어 회화에 맞는 영어 회화를 작성해 주세요.
양식은 [FORMAT]을 참고하여 작성해 주세요.

# 상황:
{question}

# FORMAT:
- 영어회화 :
- 한글번역 :
'''

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip() # strip() 공백이나 줄바꿈을 무시 (오탈자로 인한 에러 방지용)
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = PromptTemplate.from_template(template=template)

model = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url, # 사제 키값을 사용할 때는 key와 URL을 함께 넣어줘야 함
)

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

chain = prompt | model | output_parser

input = {'question', '저는 부산에서 물밀면을 먹고 싶어요.'}
response = chain.invoke(input) 
# print(response.content)      # output_parser 가 있으면 필요 없음. '.content'를 하면 에러 발생
print(response)