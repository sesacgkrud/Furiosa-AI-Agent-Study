# Decorator : 기존 함수를 수정하지 않고, 함수에 새로운 기능을 추가하는 문법

# [목적]
# 데코레이터(decorator) 의 기본 구조를 만들어, 함수 실행 전 · 후에 문장을 출력하는 기능을 덧붙여 본다.
#
# [구조]
#   def deco(func):          ← 꾸밀 함수(func) 를 인자로 받는다 (파이썬은 함수도 값처럼 넘길 수 있다)
#       def wrapper():       ← 안쪽 함수 : func 실행 앞뒤에 추가 기능을 넣는다
#           ...
#           func()
#           ...
#       return wrapper       ← 실행 결과가 아니라 wrapper 함수 자체를 돌려준다 (괄호 없음)
#
#   @deco                    ← 함수 정의 바로 위에 붙이면
#   def hello2(): ...           hello2 = deco(hello2) 와 같다 → 이후 hello2() 는 wrapper() 가 실행된다
#
# [흐름]
#   1. hello() : 데코레이터 없이 그대로 실행 → "안녕하세요."
#   2. hello2() : @deco 가 붙어 "함수 실행 전" → "안녕하세요." → "함수 실행 후"
#      hello2 안의 코드는 hello 와 같지만, 고치지 않고 앞뒤 기능만 추가됐다

def hello():
    print("안녕하세요.")

hello()
# 안녕하세요.

print("==================== 1 ====================")

def deco(func):
    def wrapper():
        print("함수 실행 전")
        func()
        print("함수 실행 후")
    return wrapper

@deco
def hello2():
    print("안녕하세요.")

hello2()
# ==================== 1 ====================
# 함수 실행 전
# 안녕하세요.
# 함수 실행 후
