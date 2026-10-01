# keras35_gpu_test00.py
# 텐서플로가 GPU 를 잡고 있는지 확인하는 파일
# list_physical_devices('GPU') 가 빈 리스트면 CPU 로만 돌아간다

import tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus) # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

if(gpus):
    print('GPU 실행!')
else:
    print('GPU 실행X')