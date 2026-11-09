# [목적] 서버 없이 파이썬 코드 안에서 furiosa_llm.LLM 으로 RNGD 에 모델을 직접 올려 답을 생성 (Day43)
# [흐름] 1. LLM(모델 이름) 으로 NPU 에 모델 로드 (with 문 → 끝나면 장치 반납)
#        2. SamplingParams 로 생성 옵션 지정
#        3. 토크나이저의 chat template 으로 메시지를 모델 입력 문자열로 변환 → llm.generate
# ※ NPU 가 하나라 furiosa-llm serve 가 떠 있으면 'No available devices' 로 실행되지 않는다

from furiosa_llm import LLM, SamplingParams

# pip install furiosa-llm

# with 블록 안에서만 모델이 NPU 에 올라가 있고, 블록이 끝나면 자동으로 내려간다
with LLM("furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic") as llm:
    # min_tokens 10 이상 · max_tokens 1000 이하로 생성, top_p 0.3 = 누적 확률 상위 30% 후보, top_k 100 = 확률 상위 100 개 후보
    params = SamplingParams(min_tokens=10, max_tokens=1000, top_p=0.3, top_k=100)
    messages = [{"role" : "user", "content" : "오늘 엔비디아 주가가 어떻게 돼?"}]
    # 메시지 리스트 → 모델이 학습한 대화 형식의 문자열로 변환 (tokenize=False : 토큰 번호가 아닌 문자열로 받음)
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False)

    # 프롬프트 리스트를 넣으면 프롬프트마다 결과가 리스트로 나온다
    response = llm.generate([prompt], params)

    # 첫 번째 프롬프트의 첫 번째 생성 결과 텍스트 (실시간 주가는 모른다고 답함)
    print(response[0].outputs[0].text)