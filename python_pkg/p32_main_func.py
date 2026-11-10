# [목적]
# 모듈마다 변수 공간(이름 공간)이 따로 있다는 것을 확인한다
# - 이 파일의 x = 222 와 p31_sample 의 x = 111 은 이름은 같아도 서로 다른 변수
# - p31_sample.test() 는 자기 모듈의 x (111) 를 출력한다

import p31_sample

x = 222
def main_func():
    print("x : ", x)

main_func()
# x :  222

p31_sample.test()
# x :  111
