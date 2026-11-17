# pd06_loc_iloc2_range.py 베이스

# [조건으로 행 고르기 (불리언 인덱싱)]
#   - 시가 · 종가를 문자열("1000") 이 아니라 숫자(1000) 로 넣었다 → >= 같은 크기 비교를 하려면 숫자여야 한다
#   - df["시가"] >= 1100 : 행마다 True / False 가 든 Series (cond1) → df[cond1] 은 True 인 행만 남긴다
#   - df[조건] 과 df.loc[조건] 은 같은 결과, iloc 은 정수 위치만 받아서 인덱스가 붙은 Series 조건은 ValueError
#   - 조건으로 고른 결과도 DataFrame 이라 ["종가"] 로 열을 고른다 (.loc["종가"] 는 행 이름으로 찾아서 KeyError)

import pandas as pd
print(pd.__version__)

data = [
    ["삼성", 1000, 2000],
    ["현대", 1100, 3000],
    ["LG", 2000, 500],
    ["아모레", 3500, 6000],
    ["네이버", 100, 1500],
]

index = ["031", "059", "033", "045", "023"]
columns = ["종목명", "시가", "종가"]

df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)

print("========== 시가가 1100원 이상인 행을 모두 출력!! ==========")
cond1 = df["시가"] >= 1100
print(cond1)
# 031    False
# 059     True
# 033     True
# 045     True
# 023    False
# Name: 시가, dtype: bool
print(df[cond1])
print(df[df["시가"] >= 1100])
#      종목명    시가    종가
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000

print("==============================")

print(df.loc[df["시가"] >= 1100])     # df[조건] 과 같은 3 개 행
# print(df.iloc[df["시가"] >= 1100]) # ValueError: iLocation based boolean indexing cannot use an indexable as a mask

#################### 종가만 출력 ####################
print("==============================")
print(df[df["시가"] >= 1100]["종가"])
# 059    3000
# 033     500
# 045    6000
# Name: 종가, dtype: int64
# print(df[df["시가"] >= 1100].loc["종가"]) # raise KeyError(key) from err
