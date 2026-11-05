# p41_import_module.py 베이스

# [목적]
# 패키지 안의 모듈을 from 으로 가져오는 두 가지 방법
# - from machine.car import drive : 함수만 가져와서 drive() 로 호출
# - from machine import car, tv    : 모듈을 가져와서 car.drive() 로 호출 (여러 개는 쉼표로)

from machine.car import drive
from machine.tv import watch

drive()
watch()

print("==========")
# from machine import car 
# from machine import tv
from machine import car, tv

car.drive()
tv.watch()

# machine 운전하다
# machine 시청하다
# ==========
# machine 운전하다
# machine 시청하다
