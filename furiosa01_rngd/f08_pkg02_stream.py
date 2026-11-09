# [목적] get_llm() 기본값 그대로 스트리밍 출력 (Day45, Day47 import 경로 변경)
# ※ import 경로 'furiosa01.a_kingkeras' 는 f08_pkg01_invoke.py 머리 주석 참고 (furiosa01 폴더 기준)

from furiosa01.a_kingkeras import get_llm

# 인자 없이 부르면 a_kingkeras.py 에 적힌 기본값(max_tokens=1000, temperature=0, top_p=0.9) 사용
llm = get_llm()

# 생성되는 대로 조각을 받아 바로 출력
for chunk in llm.stream("NPU와 GPU의 차이점을 설명해줘"):
    print(chunk.content, end="", flush=True)