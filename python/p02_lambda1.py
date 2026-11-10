# [목적]
# def 로 만든 함수와 같은 기능을 lambda 한 줄로 만들어 결과를 비교한다
# - lambda 매개변수 : 돌려줄 값  → 이름 없는 한 줄 함수 (return 을 쓰지 않아도 : 뒤의 값이 돌려진다)
# - 변수에 담으면(aaa = lambda ...) 일반 함수처럼 aaa(10) 으로 호출할 수 있다

def multiply(x):
    return x * 2

print(multiply(10))              # 20
print("====================")

aaa = lambda x : x * 2           # def multiply(x): return x * 2 와 같은 함수
print(aaa(10))                   # 20
