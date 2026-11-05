# [목적]
# 하이퍼파라미터 튜닝을 이중 for 문으로 직접 해 본다 (GridSearchCV 의 원리)
# - learning_rate 6 개 × max_depth 6 개 = 36 가지 조합으로 HistGradientBoostingClassifier 를 각각 훈련 · 평가한다
# - enumerate 로 (index, 값) 을 같이 꺼내 몇 번째 조합인지 출력한다
# - 지금까지의 최고 점수보다 높으면 best_score · best_parameters 를 바꿔 저장한다
#   → best_parameters = lr, md 는 (lr, md) 튜플로 저장된다
# - "{:.2f}".format(값) : 소수점 둘째 자리까지 출력

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001, 0.3]
max_depth = [3, 4, 5, 6, 7, 8]
best_score = 0             # 처음 비교 기준 (어떤 점수든 0 보다 크면 바뀐다)
best_parameters = ''

for i, lr in enumerate(learning_rate): # i = index, lr = value
    for j, md in enumerate(max_depth):
        model = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_depth=md,
        )
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)        # 분류 : accuracy

        if score > best_score:
            best_score = score
            best_parameters = lr, md
        print(">>>>>>>>>>", i, ",", j, "번째 score : ", round(score, 4), "도는 중... >>>>>>>>>>")

print("최고 점수 : {:.2f}".format(best_score))
print("최적 매개변수 : ", best_parameters)

# 최고 점수 : 0.97
# 최적 매개변수 :  (0.3, 4)
# → 36 개 중 learning_rate=0.3, max_depth=4 (5 , 1 번째) 가 score 0.9694 로 가장 높다
# → test 한 번 평가로 고른 값이라, 교차 검증을 같이 쓰는 GridSearchCV 로 이어진다 (m08)