# p02_tv.py 베이스

# [목적]
# p11_car 와 같이 직접 실행할 때의 __name__ ("__main__") 을 출력한다

def watch():
    print("시청하다")

watch()

print("p12_tv.py의 module 이름은 : ", __name__)
# 시청하다
# p12_tv.py의 module 이름은 :  __main__
