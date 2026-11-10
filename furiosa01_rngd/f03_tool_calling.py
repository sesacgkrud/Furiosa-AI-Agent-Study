# [목적] furiosa-llm 서버(Qwen3-8B)에 tools 를 넘겨 모델이 호출할 함수와 인자(JSON)를 고르게 하는 Tool calling 실습 (Day43)
# [흐름] 1. 실제 파이썬 함수 get_weather · get_time 정의
#        2. tools : 함수 이름 · 설명 · 인자 형식(JSON Schema) 을 모델에게 알려 주는 목록
#        3. tool_choice 로 get_weather 를 반드시 호출하게 지정해 요청
#        4. 모델이 돌려준 함수 이름 · 인자(JSON 문자열) → json.loads → 실제 함수 실행
# ※ 아래 serve 메모의 --devices npu:1 은 장치가 npu:0 하나뿐인 지금 환경에서는 npu:0 으로 띄워야 한다

"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
    --devices npu:0 \
    --host 0.0.0.0 \
    --port 8000
"""

import json
import os
from openai import OpenAI

# 환경 변수가 있으면 그 값, 없으면 로컬 furiosa-llm 서버 주소를 쓴다
base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
client = OpenAI(base_url=base_url, api_key=api_key)

# 모델이 고른 인자로 실제 실행할 함수 (실습용이라 날씨를 조회하지 않고 문장만 돌려준다)
def get_weather(location:str, unit:str):
    return f"Getting the weather for {location} in {unit}"

def get_time(timezone:str):
    return f"Getting the current time in {timezone} ..."

# 모델에게 '이런 함수들을 쓸 수 있다' 고 알려 주는 도구 목록
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            # parameters : 함수 인자의 이름 · 타입 · 설명 (JSON Schema 형식)
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                    # enum : 이 두 값 중 하나만 고르게 한다
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                # required : 반드시 채워야 하는 인자
                "required": ["location", "unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a given timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone, e.g., 'America/Los_Angeles'"}
                },
                "required": ["timezone"]
            }
        }
    }
]

response = client.chat.completions.create(
    # 서버에 떠 있는 모델 이름
    model = client.models.list().data[0].id,
    messages = [{"role" : "user", "content" : "오늘 판교의 날씨는 어때?"}],
    tools = tools,
    # auto 대신 get_weather 를 반드시 호출하도록 강제
    tool_choice = {"type" : "function", "function" : {"name" : "get_weather"}},
    # 0 → 매번 같은 결과
    temperature = 0.0,
)

# 모델은 함수를 직접 실행하지 않고 '호출할 함수 이름 + 인자' 만 돌려준다
tool_call = response.choices[0].message.tool_calls[0].function
print(f"function called : {tool_call.name}")
print(f"Arguments : {tool_call.arguments}")
# 고른 함수가 get_weather 가 아니면 AssertionError 로 멈춘다
assert tool_call.name == "get_weather", f"Expected 'get_weather' but got '{tool_call.name}'"
# arguments 는 JSON 문자열 → json.loads 로 dict → ** 로 풀어서 get_weather(location=..., unit=...) 실행
print(f"Result : {get_weather(**json.loads(tool_call.arguments))}")

# 실행 결과
# function called : get_weather
# Arguments : {
#   "location": "판교, 한국",
#   "unit": "celsius"
# }
# Result : Getting the weather for 판교, 한국 in celsius