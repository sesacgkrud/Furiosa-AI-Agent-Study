# [목적]
# LangGraph State 의 messages 에 쓰는 Reducer, add_messages 의 동작을 확인한다
#
# - Annotated[list, add_messages] : messages 는 list 이고, 새 값이 들어오면 덮어쓰지 않고 add_messages 로 합친다
# - add_messages(기존 리스트, 새 리스트) : 두 메시지 리스트를 이어 붙인다
#                                         단, id 가 같은 메시지는 추가하지 않고 새 메시지로 바꾼다

from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class MyData(TypedDict):
    messages : Annotated[list, add_messages] # Reducer

from langchain_core.messages import AIMessage, HumanMessage

msg1 = [HumanMessage(content='안녕', id='1')]
msg2 = [AIMessage(content='방가', id='2')]

results = add_messages(msg1, msg2) # msg1, msg2 리스트 형태로 들어감
print(results)
# [HumanMessage(content='안녕', additional_kwargs={}, response_metadata={}, id='1'),
# AIMessage(content='방가', additional_kwargs={}, response_metadata={}, id='2', tool_calls=[], invalid_tool_calls=[])]

##################################################

msg3 = [HumanMessage(content='안녕', id='3')]
msg4 = [AIMessage(content='방가', id='3')]

results2 = add_messages(msg3, msg4) # msg3, msg4 리스트 형태로 들어감
print(results2)                     # add_messages 사용하더라도 id가 동일하면 덮어쓰게 된다
# [AIMessage(content='방가', additional_kwargs={}, response_metadata={}, id='3', tool_calls=[], invalid_tool_calls=[])]
