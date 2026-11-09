# [목적] PromptTemplate 으로 {변수} 자리가 있는 프롬프트 틀 만들기 (Day19)
# [흐름] 템플릿 문자열 → PromptTemplate.from_template → 객체 출력 (input_variables 에 country 가 잡힌다)
# ※ 맨 위 메모 : 체인 순서 prompt → model(llm) → output(parser)

# pmo : prompt - model - output
# plp : prompt - llm - parser

from langchain_core.prompts import PromptTemplate

template = '{country}의 수도는 어디인가요?'

# {country} 를 입력 변수로 인식 → 나중에 invoke({'country' : '한국'}) 처럼 값을 채운다
prompt_template = PromptTemplate.from_template(template)
print(prompt_template) # input_variables=['country'] input_types={} partial_variables={} template='{country}의 수도는 어디인가요?'