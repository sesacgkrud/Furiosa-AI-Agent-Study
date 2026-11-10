# p03_do.py 베이스

# [목적]
# p11_car · p12_tv 를 import 했을 때 각 모듈의 __name__ 이 파일 이름으로 바뀌는 것을 확인한다
# - 직접 실행한 이 파일의 __name__ 만 "__main__"

import p11_car
import p12_tv

# 운전하다
# p11_car.py의 module 이름은 :  p11_car
# 시청하다
# p12_tv.py의 module 이름은 :  p12_tv

print("==========")
print("p13_do.py의 module 이름은 : ", __name__)
print("==========")

p11_car.drive()
p12_tv.watch()

# ==========
# p13_do.py의 module 이름은 :  __main__
# ==========
# 운전하다
# 시청하다
