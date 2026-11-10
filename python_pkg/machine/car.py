# [목적]
# 패키지(machine 폴더) 안의 모듈 : p41 · p42 · p51 에서 machine.car 로 불러 쓴다
# - drive() 호출을 if __name__ == "__main__": 안에만 둬서 import 할 때는 아무것도 실행되지 않는다

def drive():
    print("machine 운전하다")

if __name__ == "__main__":
    drive()
