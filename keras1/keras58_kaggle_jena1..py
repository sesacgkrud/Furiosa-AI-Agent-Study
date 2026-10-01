# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
# 독일 예나(Jena) 기상 관측 데이터 : 2009.01.01 00:10 ~ 2017.01.01 00:00, 10분 간격, 14개 컬럼

# [실습]
#   y 는 'wd (deg)' (풍향), 자르는 건 마음대로 (y : 144개)
#   [수정] y 를 'T (degC)' (기온) 로 바꿔서 실습한다
#   맞추기 : 2016.12.31 00:10 ~ 2017.01.01 00:00 데이터 144개 (10분 × 144 = 24시간)
#            → 이 144개는 훈련에 사용하지 않는다
#
# jena1 : csv 를 읽어서 x / y / 정답 구간을 잘라 npy 로 저장 (한 번만 실행)
# jena2 : npy 를 불러와서 split_x → 스케일링 → 훈련 → 예측
#   csv 를 매번 읽는 것보다 npy 로 저장해두고 불러오는 게 훨씬 빠르다

import os
import numpy as np
import pandas as pd

#1. 데이터
path = './_data/kaggle_jena/'
# index_col=0 : 첫 번째 컬럼 'Date Time' 을 인덱스로 → 나머지 14개 컬럼만 데이터로 남는다
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

print(datasets.shape)       # (420551, 14)
print(datasets.columns)
# ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)',
#  'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)',
#  'wv (m/s)', 'max. wv (m/s)', 'wd (deg)']
print(datasets.isna().sum().sum())  # 0 → 결측치 없음

# 전체를 뒤에서부터 세 구간으로 나눈다 (전체 행 개수 N = 420551)
#
#   [0 ................................ N-288]  [N-288 ~ N-144]  [N-144 ~ N]
#   └──────── 훈련에 쓰는 구간 ────────┘  └ 예측용 x 144개 ┘  └ 정답 y 144개 ┘
#
# "직전 144개(하루) 를 보고 → 다음 144개(하루) 의 풍향을 맞춘다" 로 설계
#   [수정] 풍향 → 기온 으로 바뀔 뿐, 구간을 나누는 방법(288 / 144) 은 그대로 쓴다
#   예측 때 : 12.30 00:10 ~ 12.31 00:00 (144개) 를 보고 → 12.31 00:10 ~ 01.01 00:00 (144개) 를 맞춘다

# 정답 : 마지막 144개의 풍향 (훈련에는 절대 쓰지 않고, 맨 마지막 채점에만 쓴다)
# y_cor = datasets[-144:]['wd (deg)']
#   [주석 이유] 정답 컬럼이 풍향(wd) 에서 기온(T) 으로 바뀌었다 → 마지막 144개의 기온을 정답으로 잡는다
y_cor = datasets[-144:]['T (degC)']
print(y_cor.shape)          # (144,)

# 예측용 x : 정답 바로 앞 144개 (y 컬럼은 빼고 13개 컬럼)
# x_predict = datasets[-288:-144].drop(['wd (deg)'], axis=1)
#   [주석 이유] y 컬럼이 T 로 바뀌었으니 x 에서 빼야 하는 컬럼도 T 로 바꿔야 한다
#               wd 를 그대로 빼면 x 에 T(=y) 가 남고, 풍향 정보는 쓸데없이 버려진다
#               → 이제 wd 는 x 의 feature 중 하나로 들어간다 (컬럼 수는 똑같이 13개)
x_predict = datasets[-288:-144].drop(['T (degC)'], axis=1)
print(x_predict.shape)      # (144, 13)

# 훈련용 x / y
#   x_data : 처음 ~ 뒤에서 288개 전까지 (예측용 x, 정답 구간은 빼야 훈련에 안 섞인다)
#   y_data : x_data 보다 144칸 뒤에서 시작 → x 의 i 번째 144개 묶음과 y 의 i 번째 144개 묶음이
#            "하루 전 → 다음 하루" 로 딱 맞게 짝지어진다
#            (x 묶음 i : i ~ i+143 행 / y 묶음 i : i+144 ~ i+287 행)
# x_data = datasets[:-288].drop(['wd (deg)'], axis=1)
# y_data = datasets[144:-144]['wd (deg)']
#   [주석 이유] x_predict / y_cor 와 같은 이유 → x 에서는 T 를 빼고, y 는 T 컬럼으로 잡는다
#               자르는 구간([:-288], [144:-144]) 은 그대로라 shape 도 그대로다
x_data = datasets[:-288].drop(['T (degC)'], axis=1)
y_data = datasets[144:-144]['T (degC)']
print(x_data.shape)         # (420263, 13)
print(y_data.shape)         # (420263,)  → x_data 와 길이가 같아야 split_x 후 개수가 맞는다

# npy 로 저장
#   float32 로 바꾸는 이유 : 기본 float64 의 절반 용량 (jena2 에서 split_x 하면 수 GB 가 되기 때문)
#   _data/*_npy/ 는 .gitignore 에 있어서 GitHub 에는 올라가지 않는다
np_path = './_data/kaggle_jena_npy/'
os.makedirs(np_path, exist_ok=True)     # 폴더가 없으면 만든다

np.save(np_path + 'keras58_x_data.npy', arr=x_data.values.astype(np.float32))
np.save(np_path + 'keras58_y_data.npy', arr=y_data.values.astype(np.float32))
np.save(np_path + 'keras58_x_predict.npy', arr=x_predict.values.astype(np.float32))
np.save(np_path + 'keras58_y_cor.npy', arr=y_cor.values.astype(np.float32))

print('npy 저장 완료')
