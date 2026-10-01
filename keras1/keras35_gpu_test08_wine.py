# keras35_gpu_test08_wine.py
# keras34 와 코드는 그대로 두고 CPU / GPU 로 각각 실행해서 소요 시간을 비교하는 파일
# 파일 아래쪽에 CPU 기록 / GPU 기록을 따로 남긴다
# 데이터가 작으면 CPU 가 더 빠를 수 있다 (GPU 로 데이터를 보내는 시간이 계산 시간보다 크기 때문)
# --- 아래는 원본(keras34_hamsu08_wine.py) 의 설명 ---
# 구조 : keras34_hamsu00.py (함수형 모델) 기준 / 데이터, 모델 : keras33_dropout08_wine.py 를 함수형으로 변환
# 함수형 모델 : Input 으로 입력층을 만들고, '층(이전 층)' 형태로 하나씩 연결한 뒤 Model(inputs, outputs) 로 범위를 정한다
#               층 구성이 같으면 Sequential 과 파라미터 수도 같은 똑같은 모델이다 (만드는 방법만 다르다)
#               단, 같은 것은 '구조(Total params)' 다. 가중치 초기값 / Dropout 이 끄는 노드 / 배치 섞기가 매번 랜덤이라
#               시드를 고정하지 않으면 같은 코드를 두 번 돌려도 loss, acc 는 조금씩 다르게 나온다
#               (random_state 는 train_test_split 의 데이터 분할만 고정한다)
# Dropout : 훈련할 때마다 층 출력의 일부 노드를 랜덤으로 꺼서 특정 노드에만 의존하지 않게 한다 -> 과적합 방지
#           evaluate / predict 때는 자동으로 꺼지고 모든 노드를 다 사용한다

import numpy as np
import time
from sklearn.datasets import load_wine     # 와인 데이터 (다중 분류 : 0 / 1 / 2)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Model                   # Model -> 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input   # Input -> 함수형 모델의 입력층
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

path = './_save/keras34/'   # 모델 저장 폴더 (furiosa_study 폴더에서 실행하는 기준)

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)

y = to_categorical(y)   # 원핫 : (178,) -> (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=50,
    shuffle=True,
    stratify=y,     # 클래스 비율을 train 과 test 에 똑같이 나눈다
)

scaler = RobustScaler()                 # keras28 에서 4종 비교 후 RobustScaler 로 통일
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

print('Min :', np.min(x_train), 'Max :', np.max(x_train))
print('Min :', np.min(x_test), 'Max :', np.max(x_test))

#2. 모델 구성 (함수형) - keras33 과 층 / Dropout 위치 / 출력층 activation 이 똑같다
# Dropout 은 keras33 과 똑같이 200 뒤 0.4, 100 뒤 0.2 두 개만 넣는다
input1 = Input(shape=(13,))                     # 입력층 : 컬럼 13개 (Sequential 의 input_dim=13)
dense1 = Dense(200, activation='relu')(input1)  # (input1) -> input1 뒤에 연결
drop1 = Dropout(0.4)(dense1)                    # 가장 큰 층(200) 출력의 40% 를 훈련 때마다 끈다

dense2 = Dense(150, activation='relu')(drop1)
dense3 = Dense(100, activation='relu')(dense2)  # Dropout 이 없는 층은 바로 앞 Dense 에 연결
drop2 = Dropout(0.2)(dense3)                    # 100 층 출력의 20% 를 끈다

dense4 = Dense(50, activation='relu')(drop2)
dense5 = Dense(10, activation='relu')(dense4)

# 다중 분류(원핫) 출력층은 softmax 가 반드시 있어야 칸별 확률(합 = 1)이 나온다
# softmax 가 없으면 출력이 확률이 아닌 실수 값 그대로 나온다
output1 = Dense(3, activation='softmax')(dense5)

model = Model(inputs=input1, outputs=output1)   # 시작(input1) ~ 끝(output1) 범위를 정해서 모델 완성
model.summary()                                 # keras33 Sequential 과 Total params 가 같아야 제대로 변환한 것

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     patience=50,                # val_loss 가 50 epoch 동안 안 좋아지면 멈춘다
#     restore_best_weights=True,  # 멈춘 뒤 val_loss 가 가장 낮았던 가중치로 되돌린다
#     verbose=1,
# )

mcp = ModelCheckpoint(
    monitor='val_loss',         # es 와 같은 기준으로 '최고 epoch' 를 고른다
    mode='auto',                # val_loss 는 낮을수록 좋으므로 auto(=min)
    save_best_only=True,        # 최고 기록이 갱신될 때만 덮어쓴다 -> 마지막에 남는 파일 = 최고 epoch 모델
    filepath=path + 'keras34_mcp8.keras',   # 파일 번호 규칙 : 08 -> mcp8
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=100,
                 batch_size=32,
                 validation_split=0.3,
                 callbacks=[mcp],   # callbacks 에 넣어야 실제로 동작한다
                 verbose=1,
                 )

end_time = time.time()

print("소요 시간 :", round(end_time - start_time, 2), "초")

print("========== ========== ========== ========== ==========")

#4. 평가 예측 (훈련에도 검증에도 안 쓴 x_test 로만)
result = model.evaluate(x_test, y_test) # [loss, acc]
print('loss :', result[0])
print('acc :', round(result[1], 3))

y_predict = model.predict(x_test)       # softmax 확률 (행마다 3칸)

# 분류 성능은 acc 로 판단한다. r2 / mse / rmse 는 '확률 vs 원핫 정답' 의 오차로 참고용이다.
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

y_test_arg = np.argmax(y_test, axis=1)          # 원핫 정답 -> 클래스 번호
y_predict_arg = np.argmax(y_predict, axis=1)    # 확률이 가장 큰 칸 -> 예측 클래스 번호

# 변수 이름을 accuracy_score 로 하면 함수를 덮어써서 다음에 또 부를 때 에러가 난다 -> acc_score 로
acc_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', acc_score)

# ===== CPU 기록 =====
# 소요 시간 : 8.02 초
# loss : 0.1516449749469757
# acc : 0.963
# r2 : 0.9148545485493251
# mse : 0.01852531937049567
# RMSE : 0.13610774911993684
# accuracy_score : 0.9629629629629629

# ===== GPU 기록 =====
# 소요 시간 : 4.65 초
# loss : 0.0985560491681099
# acc : 0.981
# r2 : 0.9433357119560242
# mse : 0.01232890784740448
# RMSE : 0.111035615220543
# accuracy_score : 0.9814814814814815