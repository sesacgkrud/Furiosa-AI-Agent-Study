# [목적]
# map 과 lambda 를 함께 써서 점수 리스트를 한 줄로 "합격" / "불합격" 리스트로 바꾼다
# - lambda x: "합격" if x >= 60 else "불합격" → 60 점 이상이면 "합격", 아니면 "불합격" 을 돌려주는 한 줄 함수
#   (값1 if 조건 else 값2 : 조건이 참이면 값1, 거짓이면 값2)
# - map(lambda ..., score) 로 score 의 각 점수에 적용하고, list() 로 감싸 결과를 리스트로 만든다
# - 따로 def 함수를 만들지 않고, 한 번만 쓸 함수를 map 안에 바로 넣을 때 lambda 를 쓴다

score = [85, 45, 70, 55, 95]
results = list(map(lambda x: "합격" if x >=60 else "불합격", score))
print(results) # ['합격', '불합격', '합격', '불합격', '합격']
