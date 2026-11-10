# [목적]
# map(함수, 리스트) 로 리스트의 각 값에 같은 함수를 한 번에 적용한다
# - map 은 결과를 바로 리스트로 주지 않고 map 객체(값을 하나씩 꺼내 쓰는 묶음) 를 돌려준다
#   → print(result) 를 하면 값이 아니라 <map object at 0x...> 처럼 객체 정보만 출력된다
# - list(result) 로 감싸야 [10, 20, 30, 40] 처럼 결과 값을 한꺼번에 볼 수 있다
# - map 객체는 한 번 꺼내 쓰면 비워진다 → list(result) 를 한 번 더 하면 [] 가 나온다

def multiply(x):                # 값 하나를 받아 10 배로 돌려주는 함수
    return x * 10

numbers = [1,2,3,4]
result = map(multiply, numbers) # numbers 의 각 값에 multiply 를 적용 (함수 이름만 넘긴다, multiply() 처럼 괄호 X)

print(result)          # <map object at 0x000001F6794BACB0>
print(list(result))    # [10, 20, 30, 40]
