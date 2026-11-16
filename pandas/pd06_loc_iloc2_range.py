# pd06_loc_iloc1.py 베이스

# [범위(슬라이싱) 로 여러 행 꺼내기]
#   - iloc[3:5] : 위치 3, 4 (끝 5 미포함) → 아모레 · 네이버
#   - loc["045":"023"] : 이름 "045" 부터 "023" 까지 (끝 포함) → 같은 두 행
#   - iloc 은 정수만, loc 은 이름만 받는다 → iloc[3:5, "시가"] (ValueError), iloc["045":"023", 1] (TypeError)
#   - 여러 행을 꺼낸 결과는 DataFrame 이라 ["시가"] 로 열을 고를 수 있고, .iloc[1] 은 그 안의 두 번째 행 (네이버)
#   - .loc["시가"] 는 행 이름으로 찾기 때문에 KeyError → 열은 ["시가"] 나 loc[행, "시가"] 로 고른다

import pandas as pd
print(pd.__version__)

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

print("========== 아모레와 네이버의 시가 ==========")
# print(df.iloc[3:5][1]) # raise KeyError(key) from err
print(df.iloc[3:5, 1])
# ========== 아모레와 네이버의 시가 ==========
# 045    3500
# 023     100
# Name: 시가, dtype: str

print(df.iloc[3:5]["시가"])
# 045    3500
# 023     100
# Name: 시가, dtype: str

# print(df.iloc[3:5, "시가"]) # ValueError: Location based indexing can only have [integer, ...] types


# print(df.iloc["045":"023"][1]) # TypeError: cannot do positional indexing on Index with these indexers [045] of type str
# print(df.iloc["045":"023", 1]) # TypeError: cannot do positional indexing on Index with these indexers [045] of type str
print(df.loc["045":"023"]["시가"])
print(df.loc["045":"023", "시가"])
# 045    3500
# 023     100
# Name: 시가, dtype: str

print("==================================================")

print(df.iloc[3:5].iloc[1])
# 종목명     네이버
# 시가      100
# 종가     1500
# Name: 023, dtype: str

# print(df.loc["045":"023"].loc["시가"]) # raise KeyError(key) from err
print(df.loc["045":"023"].iloc[1])
# 종목명     네이버
# 시가      100
# 종가     1500
# Name: 023, dtype: str
