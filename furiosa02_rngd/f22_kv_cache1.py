# [목적] 입력 길이 · 출력 길이를 바꿔 요청을 보내며 입력 / 출력 토큰 수와 응답 시간을 비교하고,
#        요청 사이에 멈춰 서버 KV Cache 사용률을 확인 (Day48)
# [흐름] 실험 3 개 (짧은 입력 · 긴 입력 · 긴 입력 + 긴 출력) → run_once → 토큰 수 · 시간 출력 → Enter 대기
# ※ KV Cache : decode 때 앞 토큰들의 Key · Value 를 다시 계산하지 않도록 NPU 메모리에 저장해 두는 공간
#   → 입력 + 출력 토큰이 많을수록 더 많이 쓴다
# ※ 결과 (실행 예) : 짧은 입력 15 토큰 / 긴 입력 1817 토큰, 출력 80 · 80 · 193 토큰, 0.89 · 0.88 · 2.14 초
#   → 입력이 120 배 길어도 응답 시간은 거의 같고(prefill 은 한 번에 처리), 출력 토큰이 늘어야 시간이 는다

from kingkeras import run_once

short_prompt = "NPU란 무엇인가?"
# 문자열 * 100 → 같은 문장을 100 번 이어 붙여 긴 입력을 만든다
long_prompt = (
    "NPU는 인공지능 연산을 가속하는 프로세서다." * 100
) + "위 내용을 요약해줘."

# (실험 이름, 질문, max_tokens) 튜플 리스트
tests = [
    ("짧은 입력", short_prompt, 80),
    ("긴 입력", long_prompt, 80),
    ("긴 입력과 긴 출력", long_prompt, 400),
]

# 튜플을 세 변수로 풀어서 받는다
for name, prompt, max_tokens in tests:
    result = run_once(prompt, max_tokens)

    print(f"\n실험: {name}")
    print("입력 토큰: ", result["input_tokens"])
    print("출력 토큰: ", result["output_tokens"])
    print("소요 시간: ", round(result["elapsed"], 2))
    # input() : Enter 를 누를 때까지 멈춘다 → 그 사이 서버 로그 · /metrics 에서 KV Cache 사용률 확인
    input("서버 KV Cache 사용률을 확인하고 싶다 Enter...")
