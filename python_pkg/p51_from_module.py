# p42_from_module.py 베이스

# [목적]
# 패키지 안의 하위 폴더(machine/test) 모듈 불러오기 : from machine.test import car, tv
# - 같은 이름(car, tv) 으로 다시 import 하면 뒤에 불러온 machine.test 의 모듈로 바뀐다

from machine.car import drive
from machine.tv import watch

drive()
watch()

print("==========")
from machine import car, tv

car.drive()
tv.watch()

print("========== test 폴더 ==========")
from machine.test import car, tv

car.drive()
tv.watch()

# machine 운전하다
# machine 시청하다
# ==========
# machine 운전하다
# machine 시청하다
# ========== test 폴더 ==========
# machine 운전하다
# machine 시청하다
