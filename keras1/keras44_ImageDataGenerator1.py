import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,           # 필수
    horizontal_flip=True,     # 수평 뒤집기 (필수x)
    vertical_flip=True,       # 수직 뒤집기 (필수x)
    width_shift_range=0.1,    # 평형 이동 (필수x)
    height_shift_range=0.1,   # (필수x)
    rotation_range=5,         # r각도 조절 (정해진 각도만큼 이미지 회전) (필수x)
    zoom_range=1.2,           # (필수x)
    shear_range=0.7,          # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동 (필수x)
    fill_mode='nearest',      # (필수x)
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test'

xy_train = train_datagen.flow_from_directory( # 해당 경로에서 가져오기 위함
    path_train,               # 경로
    target_size=(100,100),    # 설정한 크기대로 이미지를 수정
    batch_size=10,
    class_mode='binary',      # 이진 분류 (ad, normal)
    color_mode='grayscale',   # 흑백 -> 출력 형태 : (batch, x, y, 1)
    shuffle=True,
)
# 실행 결과 : Found 160 images belonging to 2 classes

xy_test = test_datagen.flow_from_directory(
    path_test,                # 경로
    target_size=(100,100),    # 설정한 크기대로 이미지를 수정
    batch_size=10,
    class_mode='binary',      # 이진 분류 (ad, normal)
    color_mode='grayscale',   # 흑백 -> 출력 형태 : (batch, x, y, 1)
    shuffle=False,            # train에서 했기 때문에 필요 없음
)
# 실행 결과 : Found 120 images belonging to 2 classes.

print(xy_train)               # <keras.preprocessing.image.DirectoryIterator object at 0x0000021ED7AE7FA0>
print(xy_train.next())        # 앞에 보여준 게 없어서 첫번째 Iterator 출력 (x와 y가 모여 있는 data)
# (array([[[[0.        ],
#          [0.        ],
#          [0.        ],
#          ...,
#          [0.        ],
#          [0.        ],
#          [0.        ]],
#          ...,
#          ...,
#          [0.        ],
#          [0.        ],
#          [0.        ]]]], dtype=float32), array([1., 0., 0., 1., 1., 0., 0., 0., 0., 0.], dtype=float32))

# print(xy_train.next())      # 두번째 Iterator 출력 (x와 y가 모여 있는 data)

# print(xy_train[0][0])       # 첫번째 batch의 x 데이터 출력
# print(xy_train[0][1])       # 첫번째 batch의 y 데이터 출력

print(xy_train[0][0].shape)   # (10, 100, 100, 1)
print(xy_train[0][1].shape)   # (10,)

# print(xy_train[16][0])      # ValueError: Asked to retrieve element 16, but the Sequence has length 16

print(type(xy_train))         # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))      # <class 'tuple'> 리스트와 비슷하지만 수정 안 됨
print(type(xy_train[0][0]))   # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))   # <class 'numpy.ndarray'>