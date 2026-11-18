# p03_Decorator.py 베이스

# [목적]
# 함수 실행 시간을 재는 timer 데코레이터를 만들어, 서로 다른 함수 두 개(hello · study) 에 @timer 하나로 똑같이 적용한다.
#
# [timer]
#   - wrapper 안에서 func() 앞뒤로 time.perf_counter() 를 찍어 (끝 - 시작) 을 걸린 시간으로 출력
#   - round(값, 6) : 소수 6 자리까지 반올림
#   - time.sleep(초) : 지정한 시간만큼 멈춤 → hello 는 약 1 초, study 는 약 1.5 초가 걸린다
#     (sleep 외에 print 등 실행 시간이 조금 더해져 1.000771 처럼 나오고, 실행할 때마다 조금씩 다르다)
#   - 시간 측정 코드를 함수마다 넣지 않고 @timer 만 붙이면 된다

# 실무에서는 다음과 같은 기능을 여러 함수에 공통으로 적용할 때 유용하다.
# - 함수 실행 시간 측정
# - 로그 기록
# - 사용자 권한 검사
# - 예외 처리
# - 함수 호출 횟수 기록

import time

def timer(func):
    def wrapper():
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        print("걸린 시간: ", round(end - start, 6), "초")
    return wrapper

@timer
def hello():
    print("안녕하세요.")
    time.sleep(1)       # 1초 간 정지

hello()
# 안녕하세요.
# 걸린 시간:  1.000771 초

@timer
def study():
    print("NPU 공부중이다.")
    time.sleep(1.5)

study()
# NPU 공부중이다.
# 걸린 시간:  1.500456 초
