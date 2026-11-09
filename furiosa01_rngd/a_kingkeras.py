# [목적] RNGD 의 furiosa-llm 서버에 붙는 LangChain ChatOpenAI 객체를 만들어 주는 공용 모듈 (Day45)
#        f08_pkg01_invoke · f08_pkg02_stream 에서 import 해서 쓴다
# [흐름] get_llm(생성 옵션) 호출 → 서버 주소 · 옵션을 넣은 ChatOpenAI 를 만들어 돌려준다

from langchain_openai import ChatOpenAI

# 생성 옵션을 매개변수로 받고 기본값을 둔다 → get_llm() 처럼 비워 부르면 기본값 사용
def get_llm(
        # max_tokens  : 답변으로 만들 최대 토큰 수
        max_tokens = 1000,
        # temperature : 0 이면 매번 확률이 가장 높은 단어를 고른다 (답이 거의 같게 나옴)
        temperature = 0,
        # top_p       : 누적 확률 상위 90% 안의 단어 후보에서만 고른다
        top_p = 0.9,
):
    llm = ChatOpenAI(                            # Request, Response 역할
        # furiosa-llm serve 가 열어 둔 OpenAI 호환 주소 (내 PC 의 8000 번 포트)
        base_url = "http://127.0.0.1:8000/v1",
        # 로컬 서버라 키 검사가 없다 → 아무 값이나 넣는다
        api_key = "EMPTY",
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
    )
    # 만든 llm 객체를 돌려준다 → 받는 쪽에서 llm.invoke / llm.stream 사용
    return llm