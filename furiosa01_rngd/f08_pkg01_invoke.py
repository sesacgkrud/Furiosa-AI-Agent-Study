# [목적] LLM 설정을 모듈(a_kingkeras.py) 의 get_llm() 으로 불러와 쓰기 — 값을 바꿔서 invoke (Day45, Day47 import 경로 변경)
# [흐름] from 패키지.모듈 import get_llm → get_llm(옵션) → llm.invoke → 답변 출력
# ※ 'furiosa01.a_kingkeras' 는 furiosa01 이라는 폴더(패키지) 안의 a_kingkeras.py 를 뜻한다.
#   지금 저장소 폴더 이름은 furiosa01_rngd 라서, 이 이름 그대로면 furiosa01 폴더가 있는 위치에서 실행해야 import 된다
#   (없으면 ModuleNotFoundError: No module named 'furiosa01')

from furiosa01.a_kingkeras import get_llm

# 기본값(1000, 0, 0.9) 대신 원하는 값만 넣어 호출
llm = get_llm(
    max_tokens = 1000,
    # 0.9 → 매번 조금씩 다른 표현의 답이 나온다
    temperature = 0.9,
    # 1.0 → 후보를 자르지 않고 전체 단어에서 선택
    top_p = 1.0,
)

response = llm.invoke("NPU와 GPU의 차이점을 설명해줘")
print(response.content)