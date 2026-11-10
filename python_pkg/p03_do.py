# [목적]
# 같은 폴더의 p01_car · p02_tv 를 import 해서 "모듈이름.함수()" 로 함수를 불러 쓴다
# - import 하는 순간 그 파일이 위에서부터 한 번 실행된다
#   → p01_car · p02_tv 맨 아래의 drive() · watch() 호출도 같이 실행돼서 "운전하다 / 시청하다" 가 먼저 찍힌다

import p01_car
import p02_tv

print("==========")
p01_car.drive()
p02_tv.watch()

# 운전하다       ← import p01_car 때 실행
# 시청하다       ← import p02_tv 때 실행
# ==========
# 운전하다       ← p01_car.drive()
# 시청하다       ← p02_tv.watch()
