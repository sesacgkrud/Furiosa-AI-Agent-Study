# [목적] 입력 길이를 늘려 가며 요청을 보내고, 요청마다 서버 /metrics 의 Prefix Cache · KV Cache 상태를 출력 (Day48)
#        f22_kv_cache1 은 Enter 로 멈춰 서버에서 직접 확인했고, 여기서는 get_cache_metrics 로 코드에서 바로 확인한다
# [흐름] 질문 3 개 (짧은 질문 · 같은 문장 × 100 · 같은 문장 × 300) → run_once(max_tokens=300)
#        → 입력 / 출력 토큰 · 응답 시간 출력 → get_cache_metrics 로 캐시 상태 출력
# ※ Prefix Cache : 이전 요청과 앞부분(prefix) 이 같은 입력이면, 그 부분의 KV 를 다시 계산하지 않고 재사용한다
#   → 질문 3 은 질문 2 의 문장으로 시작하므로 질문 2 에서 계산한 앞부분을 재사용할 수 있다
# ※ get_cache_metrics 의 값은 서버를 켠 뒤부터의 누적값 → 요청 전후의 차이로 이번 요청의 조회 · 적중 토큰 수를 본다
# ※ 결과 (실행 예, 같은 질문을 이미 보낸 적 있는 서버) : 입력 15 · 909 · 2709 토큰, 응답 시간 3.25 · 3.28 · 3.38 초
#   조회 · 적중 증가량 14 / 14 · 908 / 908 · 2708 / 2708 → 앞서 보낸 같은 질문이 캐시에 남아 있어 입력 거의 전부 적중

from kingkeras import run_once, get_cache_metrics

prompts = [
    "NPU란 무엇인가?",
    "NPU의 특징을 설명해줘." * 100,
    "NPU의 특징을 설명해줘." * 300,
]

# enumerate(리스트, 1) : (번호, 값) 을 함께 꺼낸다, 번호는 1 부터
for i, prompt in enumerate(prompts, 1):
    print(i)
    print(f"\n========== 실험 {i} ==========")
    result = run_once(prompt, 300)
    print("입력 토큰: ", result["input_tokens"])
    print("출력 토큰: ", result["output_tokens"])
    print("응답 시간: ", round(result["elapsed"], 2))

    # 요청이 끝난 뒤 호출해야 이번 요청까지 반영된 값이 나온다
    print("\nKV Cache 매트릭")
    get_cache_metrics()
