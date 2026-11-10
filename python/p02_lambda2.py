# [목적]
# 매개변수가 2 개인 lambda 를 만든다
# - lambda x, y : x + y → def aaa(x, y): return x + y 와 같다 (매개변수는 , 로 구분)

aaa = lambda x, y : x + y
print(aaa(1, 2)) # 3
print(aaa(3, 4)) # 7
