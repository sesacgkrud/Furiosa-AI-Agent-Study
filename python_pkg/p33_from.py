# p32_main_func.py 베이스

# [목적]
# from 모듈 import 함수 → 모듈 이름 없이 함수 이름만으로 바로 호출한다
# - from p31_sample import test 로 가져와도 test() 는 여전히 p31_sample 의 x (111) 를 쓴다
# - import p31_sample 과 함께 써도 결과는 같다

# import p31_sample
from p31_sample import test

x = 222
def main_func():
    print("x : ", x)

main_func()
# x :  222

test()
# x :  111

##############################

print("==========")
import p31_sample
p31_sample.test()
# x :  111
