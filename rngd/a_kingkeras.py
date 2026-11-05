from langchain_openai import ChatOpenAI

def get_llm(
        max_tokens = 1000,
        temperature = 0,
        top_p = 0.9,
):
    llm = ChatOpenAI(                            # Request, Response 역할
        base_url = "http://127.0.0.1:8000/v1",
        api_key = "EMPTY",
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
    )
    return llm