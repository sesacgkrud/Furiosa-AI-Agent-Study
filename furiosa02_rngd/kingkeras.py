# [목적] furiosa02_rngd 실습 파일들이 공통으로 import 하는 LLM 생성 모듈 (Day47)
#        Day45 의 a_kingkeras.py 와 같은 역할, 기본값을 None 으로 바꿨다
# [흐름] get_llm(max_tokens, temperature, top_p) → furiosa-llm 서버(127.0.0.1:8000) 에 붙는 ChatOpenAI 반환
# ※ None = 값을 정하지 않음 → 요청에 넣지 않아 서버(모델) 기본 설정을 따른다

from  langchain_openai import ChatOpenAI

# 세 옵션 모두 기본값 None → 필요한 것만 골라서 넘긴다 (예: get_llm(max_tokens=500))
def get_llm(max_tokens=None, temperature=None, top_p=None):
    llm = ChatOpenAI(
        # furiosa-llm serve 로 띄운 로컬 OpenAI 호환 서버
        base_url = "http://127.0.0.1:8000/v1",
        # 로컬 서버라 아무 값
        api_key = "EMPTY",
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
    )
    return llm

