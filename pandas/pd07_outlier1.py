# [목적]
# IQR(사분위 범위) 방식으로 1 차원 데이터의 이상치(outlier) 위치를 찾고, boxplot 으로 확인한다.
#
# [IQR 이상치]
#   - np.percentile(data, [25, 50, 75]) : 1사분위(Q1, 하위 25%) · 중위값(q2, 50%) · 3사분위(Q3, 75%)
#   - IQR = Q3 - Q1 (가운데 50% 데이터가 퍼진 폭)
#   - 하한 = Q1 - 1.5 * IQR, 상한 = Q3 + 1.5 * IQR → 이 범위를 벗어나면 이상치
#   - np.where(조건1 | 조건2) : 둘 중 하나라도 참인 위치(인덱스) 를 튜플로 돌려준다 (| = 또는)
#   - 함수가 값을 여러 개 return 하면 outlier_loc, iqr, low, up 처럼 한 번에 나눠 받는다 (줄 끝 \ 는 다음 줄로 이어짐)
#
# [결과]
#   Q1 4, q2 7, Q3 10, IQR 6 → 하한 -5, 상한 19 → 이상치 -10 (0 번), 50 (12 번)
#
# [boxplot]
#   상자 = Q1 ~ Q3, 가운데 선 = 중위값, 수염 = 범위 안의 최솟값 · 최댓값, 바깥 점 = 이상치
#   plt.axhline(y) : y 위치에 가로선 → 상한 · 하한을 분홍 선으로 표시

import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])

def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75])
    print("1사분위: ", quartile_1)
    print("q2: ", q2)
    print("3사분위: ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("IQR: ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data > upper_bound) | (data < lower_bound)), \
           iqr, lower_bound, upper_bound

outlier_loc, iqr, low, up = outlier(aaa)
print("이상치의 위치: ", outlier_loc)
# 1사분위:  4.0
# q2:  7.0
# 3사분위:  10.0
# IQR:  6.0
# 이상치의 위치:  (array([ 0, 12]),)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.axhline(up, color="pink", label="upper bound")
plt.axhline(low, color="pink", label ="lower bound")
plt.legend()
plt.show()
