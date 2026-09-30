# pmo : prompt - model - output
# plp : prompt - llm - parser

from langchain_core.prompts import PromptTemplate

template = '{country}의 수도는 어디인가요?'

prompt_template = PromptTemplate.from_template(template)
print(prompt_template) # input_variables=['country'] input_types={} partial_variables={} template='{country}의 수도는 어디인가요?'