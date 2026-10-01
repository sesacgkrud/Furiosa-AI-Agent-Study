'''
개, 고양이 가중치를 가져와서 모델 완성    --- kaggle에서 했던 개, 고양이 가중치 저장해온 것 load
데이터는 개, 고양이 npy 데이터 사용      --- 
남자 연예인 사진(image.png, keras48_man.npy)도 npy 불러와서 predict
'''

# keras49_01_내가고양이상인가.py
# 저장해 둔 개 / 고양이 모델로 사진 한 장이 개상인지 고양이상인지 본다
#
# [ 이 파일이 하는 일 ]
#  1) keras45_03 이 저장한 cat_dog test npy 로 모델 실력을 먼저 확인한다
#  2) keras48 이 저장한 사진 한 장(keras48_man.npy)을 같은 모델에 넣어 예측한다
#  -> 훈련은 하지 않는다. 이미 저장된 모델을 불러오기만 한다
#
# [ 모델을 불러오는 두 가지 ]
#  load_model   : 구조 + 가중치 + 컴파일 설정까지 한 파일에서 불러온다 (모델을 다시 만들 필요가 없다)
#  load_weights : 같은 구조의 모델을 먼저 만들어 둔 뒤 가중치만 덮어쓴다
#  -> ModelCheckpoint 로 저장한 .keras 파일은 전자로 불러오는 게 간단하다
#
# [ 예측값을 읽는 법 ]
#  cat_dog 은 class_mode='binary' 로 읽었고 클래스 번호는 폴더 이름 알파벳순 -> cats 0, dogs 1
#  출력층이 Dense(1, sigmoid) 라 예측값은 '정답이 1인 쪽(개)일 확률' 하나다
#  0 에 가까우면 고양이상, 1 에 가까우면 개상
#  (번호가 헷갈리면 generator 에서 class_indices 를 찍어 확인한다)

import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

#1. 데이터
########## cat_dog test 데이터 (keras45_03 이 저장한 npy) ##########
np_path = './_data/kaggle_cat_dog_npy/'
x_test = np.load(np_path + 'keras45_03_x_test.npy')
y_test = np.load(np_path + 'keras45_03_y_test.npy')

print(x_test.shape, y_test.shape)       # (2023, 100, 100, 3) (2023,)

########## 내 사진 (keras48 이 저장한 npy) ##########
x_man = np.load(np_path + 'keras48_man.npy')

print(x_man.shape)              # (1, 100, 100, 3)  <- predict 에 넣으려면 4차원이어야 한다
print(np.max(x_man))            # 255.0  <- img_to_array 결과는 0 ~ 255 그대로다

# 훈련할 때 rescale=1./255 로 0 ~ 1 을 넣었으므로 예측할 사진도 똑같이 나눠 줘야 한다
# 이 줄을 빼면 모델이 훈련 중 본 적 없는 큰 숫자가 들어가 엉뚱한 값이 나온다
x_man = x_man/255.
print(np.max(x_man))            # 1.0

#2. 모델 (훈련 없이 저장해 둔 모델을 불러온다)
# cat_dog 훈련이 끝난 뒤 저장한 모델 (test acc 0.787)
# 파일명에 val_acc 가 들어간 ModelCheckpoint 파일을 써도 된다 : k_0921_1825_0013_0.7939.keras
# 불러올 모델의 acc 가 0.5 근처면 무엇을 넣어도 반반이 나와 예측이 의미가 없다
#  -> 그때는 예측 코드가 아니라 훈련 쪽을 손봐야 한다 (keras44_ImageDataGenerator3_CatDog.py 주석 참고)
path_model = './_save/keras44/keras44_catdog.keras'
model = load_model(path_model)

model.summary()

#3. 평가 (사진을 넣기 전에 test 데이터로 이 모델이 얼마나 맞히는지 먼저 본다)
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)         # sigmoid 확률 -> 0.5 기준 반올림 (0 고양이 / 1 개)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)

#4. 내 사진 예측
print('========== 내가 고양이상인가 ==========')
y_man = model.predict(x_man)            # (1, 1) 짜리 sigmoid 확률

dog = float(y_man[0][0])                # 1 에 가까울수록 개상
cat = 1 - dog                           # 나머지가 고양이상 (두 확률의 합은 1)

print('고양이상 :', round(cat * 100, 2), '%')
print('개상 :', round(dog * 100, 2), '%')

if cat > dog:
    print('-> 고양이상 입니다')
else:
    print('-> 개상 입니다')
