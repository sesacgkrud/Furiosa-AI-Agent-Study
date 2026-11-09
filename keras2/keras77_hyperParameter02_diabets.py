# keras11_2_diabetes.py 베이스
#
# [목적] keras 신경망을 KerasRegressor 로 감싸 sklearn 의 RandomizedSearchCV 에 넣고,
#        batch_size · optimizer · dropout · activation · 층별 노드 수 조합을 무작위로 탐색 (diabetes 회귀)
# [흐름] build_model(하이퍼파라미터) 함수로 모델 생성 → create_hyperparameter() 로 후보 dict
#        → KerasRegressor(build_fn=build_model) → RandomizedSearchCV(n_iter=10, cv=3) → fit (EarlyStopping · ReduceLROnPlateau) → 최적 조합 · 점수


from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=42,
)

#2. 모델 구성
# RandomizedSearchCV 가 뽑은 조합을 인자로 받아 매번 새 모델을 만드는 함수 (인자 이름이 아래 dict 의 키와 같아야 한다)
def build_model(drop=0.5, optimizer='adam', activation='relu',
                # lr 은 받기만 하고 함수 안에서 쓰지 않는다 (optimizer 는 문자열이라 기본 learning_rate 로 만들어진다)
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(10,), name='inputs')

    # 함수형 모델 : 같은 변수 x 에 층을 이어 붙여 나간다
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # compile 까지 함수 안에서 해 둬야 KerasRegressor 가 바로 fit 할 수 있다
    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    return model

# 탐색할 후보 목록 : batch_size 5 x optimizer 3 x drop 4 x activation 4 x node 4^5 = 245,760 조합
def create_hyperparameter():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16]
    return {
        'batch_size' : batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation' : activations,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5,
    }
hyperparameters = create_hyperparameter()
print(hyperparameters)
# {'batch_size': [32, 16, 8, 1, 64], 'optimizer': ['adam', 'rmsprop', 'adadelta'],
# 'drop': [0.2, 0.3, 0.4, 0.5], 'activation': ['relu', 'elu', 'selu', 'linear'],
# 'node1': [128, 64, 32, 16], 'node2': [128, 64, 32, 16], 'node3': [128, 64, 32, 16],
# 'node4': [128, 64, 32, 16], 'node5': [128, 64, 32, 16]}

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# keras 모델 → sklearn 모델처럼 fit / score 를 쓸 수 있게 감싼다 (build_fn = 모델을 만드는 함수)
keras_model = KerasRegressor(build_fn=build_model, verbose=1,)

# 245,760 조합 중 n_iter=10 개만 무작위로 뽑아 cv=3 교차 검증 → 10 x 3 = 30 번 훈련
model = RandomizedSearchCV(keras_model, hyperparameters,
                           cv=3,
                           random_state=333,
                           n_iter=10,
                           verbose=1,
                           refit=True,
                        #    n_jobs=1,
                        )

# 3. 훈련
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,

)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=8,
    verbose=1,
    factor=0.8,
)

start = time.perf_counter()
# fit 에 준 epochs · callbacks · validation_split 은 조합마다 만들어지는 keras 모델의 fit 으로 그대로 전달된다
model.fit(x_train, y_train, epochs=1000, verbose=1, callbacks=[es, rlr],
          validation_split=0.1)
end = time.perf_counter()

print("==============================")
print("최적의 매개변수 :", model.best_estimator_)
print("==============================")
print("최적의 파라미터 :", model.best_params_)
print("==============================")

# 4. 평가, 예측
print("==============================")
# KerasRegressor 의 score 는 loss(mse) 에 - 를 붙인 값 → 0 에 가까울수록 좋다
print("model.score : ", model.score(x_test, y_test))
print("best_score : ", model.best_score_)
print("걸린시간 : ", round(end - start, 2), "초")
print("==============================")

# 결과 : test mse 약 26464 (RMSE 약 163), 교차 검증 평균 mse 약 29698
# model.score :  -26464.02734375
# best_score :  -29698.218098958332
# 걸린시간 :  7.64 초