# [목적] 미리 컴파일해 둔 모델 파일(.fxb)을 fxb 인자로 직접 지정해 LLM 을 불러오기 (Day43)
#        fxb 를 주면 모델을 다시 컴파일하지 않고 바로 NPU 에 올린다 (출력 없이 로딩만 확인)

from furiosa_llm import LLM

# 첫 인자 = 모델 이름(토크나이저 · 설정용), fxb = 컴파일된 모델 파일 경로
llm = LLM("Qwen/Qwen3-8B-FP8", fxb="/root/qwen3-8b-fp8.fxb")