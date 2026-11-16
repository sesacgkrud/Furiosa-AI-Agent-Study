# [목적]
# pandas DataFrame 에서 원하는 행 · 열 · 값을 꺼내는 방법 (df[컬럼], loc, iloc) 을 비교한다.
#
# [정리]
#   - df[...]  : 컬럼 이름이 기준 → df["종목명"] 은 열, df["종목명"]["045"] 은 "열 다음 행" 순서 (Pandas 열행)
#                df[0] · df["031"] 처럼 행 번호나 인덱스 이름을 넣으면 KeyError
#   - loc      : 인덱스 이름(라벨) 기준 → df.loc["045"], df.loc["045", "종가"]  (df.loc[3] 은 KeyError)
#   - iloc     : 정수 위치(0 부터) 기준 → df.iloc[3], df.iloc[3, 2]          (df.iloc["045"] 은 TypeError)
#   - 행 하나를 꺼내면 Series 가 되고, 그 Series 에 다시 .loc["종가"] (이름) / .iloc[2] (위치) 를 쓸 수 있다
#   - pandas 3 에서는 Series[정수] 가 위치가 아니라 이름으로만 찾는다 → df.iloc[3][2] 는 KeyError, df.iloc[3].iloc[2] 로 써야 한다
#
# [데이터] 시가 · 종가를 문자열("1000") 로 넣어서 출력 dtype 이 str 이다

import pandas as pd
print(pd.__version__) # 3.1.0

data = [
    ["삼성", "1000", "2000"],
    ["현대", "1100", "3000"],
    ["LG", "2000", "500"],
    ["아모레", "3500", "6000"],
    ["네이버", "100", "1500"],
]

index = ["031", "059", "033", "045", "023"]
columns = ["종목명", "시가", "종가"]

df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500

# print(df[0])      # raise KeyError(key) from err
# print(df["031"])  # raise KeyError(key) from err
print(df["종목명"])  # ★★★ "Pandas 열행" ★★★ <- 컬럼이 기준

########## 아모레 시가 출력 ##########
# print(df[3, 1])   # raise KeyError(key) from err
# print(df[3][1])   # raise KeyError(key) from err
# print(df["045", "종목명"]) # raise KeyError(key) from err
# print(df["045"]["종목명"]) # raise KeyError(key) from err
print(df["종목명"]["045"])   # ★★★ "Pandas 열행" ★★★ <- 컬럼이 기준

######################################################################
# loc : 인덱스 기준으로 행 데이터 추출
# iloc : 행 번호를 기준으로 행 데이터 추출
        # ★★★ int loc로 외워라!!! ★★★
######################################################################

print("========== 아모레 데이터 출력 ==========")
# print(df.iloc[3])
# print(df.iloc["045"]) # TypeError: Cannot index by location index with a non-integer key
# print(df.loc[3])      # KeyError: 3
print(df.loc["045"])
# ========== 아모레 데이터 출력 ==========
# 종목명     아모레
# 시가     3500
# 종가     6000
# Name: 045, dtype: str

print("========== 네이버 데이터 출력 ==========")
# print(df.iloc[4])
# print(df.iloc["023"]) # TypeError: Cannot index by location index with a non-integer key
# print(df.loc[4])      # KeyError: 4
print(df.loc["023"])
# ========== 네이버 데이터 출력 ==========
# 종목명    네이버
# 시가     100
# 종가     1500
# Name: 023, dtype: str

print("========== 아모레 종가 출력 ==========")
# print(df.iloc[3][2])      # raise KeyError(key) from err
print(df.iloc[3, 2])        # 6000   [행 위치, 열 위치]
print(df.iloc[3]["종가"])    # 6000
# print(df.iloc[3, "종가"])  # ValueError: Location based indexing can only have [integer, ...] types

# print(df.loc["045"][2])   # raise KeyError(key) from err
# print(df.loc["045", 2])   # raise KeyError(key) from err
print(df.loc["045"]["종가"]) # 6000
print(df.loc["045", "종가"]) # 6000   [행 이름, 열 이름]

print("==================================================")

print(df.iloc[3].iloc[2])       # 6000
print(df.iloc[3].loc["종가"])    # 6000

print(df.loc["045"].loc["종가"]) # 6000
print(df.loc["045"].iloc[2])    # 6000
