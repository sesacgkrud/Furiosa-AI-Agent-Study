# p03_do.py 베이스

# [목적]
# p21_car · p22_tv 를 import 하면 if __name__ == "__main__": 안의 호출은 실행되지 않는 것을 확인한다
# - if 밖에 있는 drive() · watch() 는 import 때 그대로 실행되고, if 안의 호출만 건너뛴다

import p21_car
import p22_tv

print("==========")
p21_car.drive()
p22_tv.watch()

# 운전하다       ← import p21_car 때 if 밖의 drive()
# 시청하다       ← import p22_tv 때 if 밖의 watch()
# ==========
# 운전하다
# 시청하다
