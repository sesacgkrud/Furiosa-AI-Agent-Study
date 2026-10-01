'''
남자, 여자 가중치를 가져와서 모델 완성    --- keras47_03 에서 훈련하고 저장한 모델 load
데이터는 남자, 여자 npy 데이터 사용       --- keras46_03 이 저장한 npy
내 사진(image.png, keras48_man.npy)도 npy 불러와서 predict
'''

# keras49_02_내가남자게여자게.py
# keras47_03 이 저장한 모델로 사진 한 장이 남자인지 여자인지 본다
#
# [ 이 파일이 하는 일 ]
#  1) keras46_03 이 저장한 man_woman test npy 로 모델 실력을 먼저 확인한다
#  2) keras48 이 저장한 사진 한 장(keras48_man.npy)을 같은 모델에 넣어 예측한다
#  -> 훈련은 하지 않는다. keras47_03 이 만들어 둔 모델을 불러오기만 한다
#
# [ 파일들의 역할 나누기 ]
#  keras46_03 : 이미지를 읽어 npy 로 저장       (느린 작업을 한 번만)
#  keras47_03 : npy 로 훈련하고 모델 저장       (여러 번 반복하며 성능을 올리는 단계)
#  keras49_02 : 저장된 모델로 예측만            (훈련 없이 바로 결과 확인)
#
# [ 예측값을 읽는 법 ]
#  man_woman 은 class_mode='binary' 로 읽었고 클래스 번호는 폴더 이름 알파벳순 -> man 0, woman 1
#  출력층이 Dense(1, sigmoid) 라 예측값은 '정답이 1인 쪽(여자)일 확률' 하나다
#  0 에 가까우면 남자, 1 에 가까우면 여자

import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

#1. 데이터
########## man_woman test 데이터 (keras46_03 이 저장한 npy) ##########
np_path = './_data/man_woman_npy/'
x_test = np.load(np_path + 'keras46_03_x_test.npy')
y_test = np.load(np_path + 'keras46_03_y_test.npy')

print(x_test.shape, y_test.shape)       # (5434, 100, 100, 3) (5434,)

########## 내 사진 (keras48 이 저장한 npy) ##########
x_me = np.load('./_data/kaggle_cat_dog_npy/keras48_man.npy')    # keras48_img_to_array.py 가 저장한 파일

print(x_me.shape)               # (1, 100, 100, 3)  <- predict 에 넣으려면 4차원이어야 한다
print(np.max(x_me))             # 255.0  <- img_to_array 결과는 0 ~ 255 그대로다

# 훈련할 때 rescale=1./255 로 0 ~ 1 을 넣었으므로 예측할 사진도 똑같이 나눠 줘야 한다
# 이 줄을 빼면 모델이 훈련 중 본 적 없는 큰 숫자가 들어가 엉뚱한 값이 나온다
x_me = x_me/255.
print(np.max(x_me))             # 1.0

#2. 모델 (훈련 없이 keras47_03 이 저장한 모델을 불러온다)
# load_model 은 구조 + 가중치 + 컴파일 설정까지 한 번에 불러온다 -> 모델을 다시 만들 필요가 없다
path_model = './_save/keras47/keras47_03_man_woman.keras'
model = load_model(path_model)

model.summary()

#3. 평가 (사진을 넣기 전에 test 데이터로 이 모델이 얼마나 맞히는지 먼저 본다)
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)         # sigmoid 확률 -> 0.5 기준 반올림 (0 남자 / 1 여자)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)

#4. 내 사진 예측
print('========== 내가 남자게 여자게 ==========')
y_me = model.predict(x_me)              # (1, 1) 짜리 sigmoid 확률

woman = float(y_me[0][0])               # 1 에 가까울수록 여자
man = 1 - woman                         # 나머지가 남자 (두 확률의 합은 1)

print('남자 :', round(man * 100, 2), '%')
print('여자 :', round(woman * 100, 2), '%')

if man > woman:
    print('-> 남자 입니다')
else:
    print('-> 여자 입니다')
