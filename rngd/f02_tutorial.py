from furiosa_llm import LLM, SamplingParams

# pip install furiosa-llm

with LLM("furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic") as llm:
    params = SamplingParams(min_tokens=10, max_tokens=1000, top_p=0.3, top_k=100)
    messages = [{"role" : "user", "content" : "오늘 엔비디아 주가가 어떻게 돼?"}]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False)

    response = llm.generate([prompt], params)

    print(response[0].outputs[0].text)