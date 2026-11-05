# [목적]
# 패키지(폴더) 안의 모듈 import : import 폴더.모듈
# - machine 폴더 안의 car.py · tv.py 를 machine.car · machine.tv 로 불러온다
# - import machine.car 로 불러오면 호출할 때도 machine.car.drive() 처럼 전체 경로를 써야 한다 (car.drive() 는 오류)

import machine.car
import machine.tv

# car.drive() # (x)

machine.car.drive()
machine.tv.watch()

# machine 운전하다
# machine 시청하다
