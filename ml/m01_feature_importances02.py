# m01_feature_importances01.py 베이스

# [목적]
# m01_01 의 분류 모델 4종을 회귀 모델(Regressor) 4종으로 바꿔 diabetes 회귀 데이터에 적용한다
# - 회귀라서 train_test_split 의 stratify 를 뺐다 (y 가 연속값이라 클래스 비율을 맞출 수 없다)
# - 회귀 모델의 model.score 는 R2(결정계수) 를 반환한다

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)             # (442, 10) (442,)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=random_state,
)

#2. 모델 구성
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]

for model in models:
    #3. 컴파일, 훈련
    model.fit(x_train, y_train)
    print('====================', model.__class__.__name__, '====================')

    #4. 평가, 예측
    score = model.score(x_test, y_test) # Regressor 기준 : R2(결정계수) 반환
    print('r2: ', score)
    print(model.feature_importances_)   # diabetes 컬럼 10개 → 중요도 10개 (모두 합하면 1)

############### 모델별 전체 실행 ###############
# ==================== DecisionTreeRegressor ====================
# r2:  0.15795709914946876
# [0.09559417 0.01904038 0.23114463 0.0534315  0.03604905 0.05879742
#  0.04902482 0.01682605 0.36525519 0.07483678]
# ==================== RandomForestRegressor ====================
# r2:  0.5265549614751442
# [0.05770917 0.01047587 0.28528549 0.09846103 0.04390962 0.05190847
#  0.05713042 0.02626033 0.28720491 0.08165469]
# ==================== GradientBoostingRegressor ====================
# r2:  0.5585049159804119
# [0.04935014 0.01077655 0.30278452 0.11174122 0.02789105 0.05764994
#  0.04058792 0.0162467  0.33840513 0.04456684]
# ==================== XGBRegressor ====================
# r2:  0.39065385219018145
# [0.04159961 0.07224615 0.17835377 0.06647415 0.04094251 0.04973729
#  0.03822911 0.10475955 0.3368922  0.07076568]
# → 4개 모델 모두 3번째(bmi) · 9번째(s5) 컬럼 중요도가 높다
