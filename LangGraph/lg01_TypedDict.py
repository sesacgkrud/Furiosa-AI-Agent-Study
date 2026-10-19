# [목적]
# LangGraph 의 State 는 TypedDict 로 정의한다 → 먼저 일반 Dict 타입 힌트와 TypedDict 를 비교해 본다
#
# 1. Dict[str, str]    : 키 · 값의 자료형을 하나로만 적을 수 있다 (모든 값이 str)
# 2. class (TypedDict) : 키마다 자료형을 따로 정한 딕셔너리 형식 (name 은 str, age 는 int)
#                        만드는 방법은 Person(키=값) 과 변수 : Person = {...} 두 가지
#                        결과는 그냥 dict 이고, 자료형은 실행할 때 검사하지 않는 '힌트' 다

from typing import Dict, TypedDict

sample_dict : Dict[str, str] = {
    'name' : '김말똥',
    'age' : '39',               # 값이 모두 str 이어야 해서 나이도 문자열
    'job' : '회사원'
}

# print(sample_dict) # {'name': '김말똥', 'age': '39', 'job': '회사원'}
# print(sample_dict['name'])

##################################################

class Person(TypedDict) :       # 키마다 자료형을 정한 딕셔너리 형식
    name : str
    age : int
    job : str

person = Person(name='홍길동', age=40, job='도적')
print(person) # {'name': '홍길동', 'age': 40, 'job': '도적'}

##################################################

person2 : Person = {'name' : '김개똥', 'age' : 20, 'job' : '전사'}    # 일반 딕셔너리에 Person 타입 힌트
print(person2) # {'name': '김개똥', 'age': 20, 'job': '전사'}
