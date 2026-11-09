# [목적] 역할 · 상황 · 출력 형식(FORMAT) 을 정한 여러 줄 템플릿으로 영어 회화 만들기 (Day19)
# [흐름] 템플릿({question}) → prompt | model | StrOutputParser → invoke → 문자열 출력
# ※ 템플릿 첫 줄 '당신의 영어를 가르치는' 은 '당신은 영어를 가르치는' 의 오타 (모델은 뜻을 알아듣고 답한다)
# ※ input 은 { 키 : 값 } 딕셔너리가 아니라 쉼표로 쓴 { 값, 값 } 집합(set) 이다
#   PromptTemplate 은 입력 변수가 1 개일 때 dict 가 아닌 값을 받으면 그 값을 통째로 변수에 넣어서 오류 없이 실행되지만,
#   프롬프트에는 "{'topic', '...'}" 처럼 집합 모양 문자열이 그대로 들어간다 (순서도 실행마다 바뀔 수 있다)
#   → 의도대로라면 {'키' : '값'} (rag08_LCEL02 처럼 콜론) 이어야 한다

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

# monorouter(사제 OpenAI 호환 서버) 키를 .env 에서 읽고 앞뒤 공백 · 줄바꿈 제거
api_key = os.environ['MONOROUTER_API_KEY'].strip() # strip() 공백이나 줄바꿈을 무시 (오탈자로 인한 에러 방지용)
base_url = 'https://monogpt.kr/api/monorouter/v1'

# from_template(template=...) : 키워드 인자로 템플릿 지정
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