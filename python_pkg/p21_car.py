# p11_car.py 베이스

# [목적]
# if __name__ == "__main__": → 이 파일을 직접 실행할 때만 안쪽 코드가 돌고, import 될 때는 돌지 않는다
# - 직접 실행 : 맨 아래 drive() + if 안의 drive() → "운전하다" 2 번
# - import (p23_main_do) : __name__ 이 "p21_car" 라서 if 안은 건너뛰고, 밖에 있는 drive() 만 1 번 실행된다
#   → import 할 때 아예 실행되지 않게 하려면 함수 호출을 if 안에만 둬야 한다 (machine/car.py)

def drive():
    print("운전하다")

drive()

if __name__ == "__main__":
    drive()

# 운전하다
# 운전하다
