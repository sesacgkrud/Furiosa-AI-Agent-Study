# p01_car.py 베이스

# [목적]
# __name__ : 파이썬이 파일(모듈)마다 넣어 주는 이름 변수
# - 이 파일을 직접 실행하면 __name__ 은 "__main__"
# - 다른 파일에서 import 되면 __name__ 은 파일 이름 "p11_car" (p13_do 에서 확인)

def drive():
    print("운전하다")

drive()

print("p11_car.py의 module 이름은 : ", __name__)
# 운전하다
# p11_car.py의 module 이름은 :  __main__
