# rag20_embedding4_ChatBot.py 베이스
# [변경] 이전 대화를 문자열({history}) 대신 메시지 리스트(HumanMessage, AIMessage) 로 넘겨서 AI 가 이전 대화를 기억하고 답한다

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import gradio as gr

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

NO_ANSWER = '주어진 정보로는 답변할 수 없습니다.'
REFUSAL_WORDS = ['수 없', '있지 않', '되지 않']

#01. 데이터 불러온다.
path = './_data/'

pdf_loader = PyPDFLoader(path + 'Attention Is All You Need.pdf')

#02. 문서를 자른다. (청킹)
pdf_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=['\n\n', '\n', ' ', '']
)

split_doc = pdf_loader.load_and_split(pdf_splitter)

#03. 문서 길이 확인
print(len(split_doc))

#04. 임베딩
embeddings = HuggingFaceEmbeddings(
    model_name='Qwen/Qwen3-Embedding-0.6B',
    model_kwargs={'device': 'cpu'},
)

#05. FAISS
# [변경] ./_db/Faiss20 에 이미 저장해 두었으므로 만들기 · 저장은 주석 처리하고 불러오기만 한다
#  → 실행할 때마다 205개 청크를 CPU 로 다시 임베딩하지 않아서 Gradio URL 이 바로 나온다
#  (PDF · chunk_size · 임베딩 모델을 바꾸면 주석을 풀고 한 번 다시 저장해야 한다)
# db = FAISS.from_documents(
#     documents=split_doc,
#     embedding=embeddings,
# )

DB_PATH = './_db/Faiss20'
# db.save_local(
#     folder_path=DB_PATH,
#     index_name='transformer_index20'
# )

db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='transformer_index20',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# [변경] similarity_score_threshold 거르기 없이, 질문과 가장 비슷한 문서 4개를 그대로 가져온다
retriever = db.as_retriever(search_kwargs={'k': 4})

#06. 답변 모델
model = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

# [추가] MessagesPlaceholder : 프롬프트 안에 메시지 리스트가 들어갈 자리 / HumanMessage, AIMessage : 사용자 · AI 메시지
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

#07. 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ('system', '''너는 Attention Is All You Need.pdf 만 보고 답하는 챗봇이다. 아래 규칙을 반드시 지켜라.
1. 컨텍스트에 질문과 관련된 내용이 있으면, 그 내용을 정리해서 답해라. 질문에 딱 맞는 문장이 없어도 관련 내용이 있으면 답한다.
2. 답변의 모든 문장은 컨텍스트에 쓰여 있는 내용으로 확인할 수 있어야 한다.
   컨텍스트에 없는 정의, 배경 설명, 숫자, 인물, 예시를 네가 아는 지식이나 일반 상식으로 덧붙이지 마라.
3. 컨텍스트에 질문과 관련된 내용이 전혀 없을 때만, 다른 말 없이 정확히 "{no_answer}" 한 문장만 출력해라.
4. 이전 대화는 "그 모델", "그 층" 같은 말이 무엇을 가리키는지 파악하는 데에만 사용해라.

이전 대화 :
{history}

컨텍스트 :
{context}'''),
    MessagesPlaceholder(variable_name='chat_history'),      # [추가] 이전 대화 메시지 리스트가 시스템 메시지와 질문 사이에 들어간다
    ('human', '{input}'),
])

# [추가] partial : {no_answer}, {history} 는 invoke 에서 넘기지 않으므로 프롬프트에 미리 채워 둔다
#  이전 대화는 MessagesPlaceholder 로 들어가므로 {history} 에는 안내 문구만 넣는다
prompt = prompt.partial(
    no_answer=NO_ANSWER,
    history='(컨텍스트 아래에 이어지는 대화 메시지 참고)',
)

#08. 체인 만들기
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever

docu_chain = create_stuff_documents_chain(model, prompt)

# [추가] 검색용 질문 바꾸기 : "그 층은?" 을 그대로 검색하면 "그 층" 이 무엇인지 몰라 문서를 못 찾는다
#  → 이전 대화를 보고 혼자서도 뜻이 통하는 질문으로 바꾼 뒤 검색 (첫 질문은 그대로 검색)
#  (rag20 은 "직전 질문 + 지금 질문" 을 이어 붙여서 검색했다)
rephrase_prompt = ChatPromptTemplate.from_messages([
    ('system', '이전 대화를 참고해서, 마지막 질문을 이전 대화 없이도 이해할 수 있는 하나의 질문으로 다시 써라. 답하지 말고 질문만 출력해라.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}'),
])
history_retriever = create_history_aware_retriever(model, retriever, rephrase_prompt)

# [추가] 검색 | docu_chain 을 하나로 묶는다 → 결과 딕셔너리의 response['answer'] 가 답변
#  (rag20 은 retriever.invoke 와 docu_chain.invoke 를 함수 안에서 따로 불렀다)
rag_chain = create_retrieval_chain(history_retriever, docu_chain)

# [삭제] #09. content_to_text, history_to_text, last_user_question
#  대화 기록을 문자열로 바꾸던 함수들 → 메시지 리스트로 넘기면서 필요 없어졌다

#10. 챗봇 함수
# [변경] Gradio history 중 최근 10개 메시지를 HumanMessage / AIMessage 리스트로 바꿔서 rag_chain 에 넘긴다
#  rag20 의 검색 결과 없음 처리 · 거절 문장 처리 · 출처 표시는 빠지고 답변만 반환한다
def answer_invoke(message, history):
    chat_history = []

    for item in history[-10:]: # [추가] 최근 10개 메시지(5번 주고받은 대화) 만 전달 → 대화가 길어져도 프롬프트 토큰이 계속 늘지 않는다
        content = item['content']

        if item['role'] == 'user':
            chat_history.append(HumanMessage(content=content))
        elif item['role'] == 'assistant':
            chat_history.append(AIMessage(content=content))    # 이 내용들이 MessagesPlaceholder(variable_name='chat_history'), 여기로 넘어감

    response = rag_chain.invoke({
        'input' : message,
        'chat_history' : chat_history,

    })

    return response['answer']

#11. Gradio 화면
demo = gr.ChatInterface(
    fn=answer_invoke,
    title='📊 Attention Is All You Need RAG 챗봇',
    # [변경] 출처 표시 대신 이전 대화를 기억한다는 설명으로 바꿈
    description='Qwen3 임베딩으로 만든 FAISS 벡터 DB 에서 Attention Is All You Need 논문만 보고 답합니다. 문서에 없는 내용은 답하지 않고, 이전 대화를 기억해서 "그 모델", "그 층" 같은 질문에도 답합니다.',
    chatbot=gr.Chatbot(
        height=500,
        placeholder='### 무엇이든 물어보세요\n아래 예시 질문을 눌러 시작할 수 있습니다.',
    ),
    textbox=gr.Textbox(placeholder='Attention Is All You Need.pdf 에 관한 내용을 물어보세요', scale=7),
    examples=[
        'Transformer 모델 구조를 설명해줘',
        'Multi-Head Attention 은 무엇이야?',
        'Positional Encoding 은 왜 필요해?',
        '삼성전자의 사업 전망은?',
    ],
)

demo.launch(theme=gr.themes.Soft())

# [추가] 대화할 때마다 Gradio history 가 쌓이는 모양
#  answer_invoke 가 불릴 때 history 에는 "지금 질문 이전" 대화만 들어 있고, 지금 질문은 message 로 따로 들어온다
'''
1. 안녕
history = [
	{'role' : 'user', 'content' : '안녕'},
	{'role' : 'assistant', 'content' : '안녕하세요'},
] # 첫번째 안녕 이라고 말했을 때 발생하는 현상 (answer_invoke 함수 -> 현재 히스토리 없음 -> rag_chain.invoke에 input -> response의 answer 리턴)
  # 답변이 끝난 뒤 질문 · 답변 2개가 history 에 추가된다

2. 나는 AI 개발자다.
history = [
	{'role' : 'user', 'content': '안녕'},
	{'role' : 'assistant', 'content' : '안녕하세요'},
	{'role' : 'user', 'content' : '나는 AI 개발자다.'},
	{'role' : 'assistant', 'content' : '당신은 AI 개발자이군요.'},
] # 이때 answer_invoke 의 history 에는 1번의 2개만 있고, '나는 AI 개발자다.' 는 message 로 들어온다

3. 내가 누구게
  # history 4개 -> HumanMessage, AIMessage, HumanMessage, AIMessage 리스트(chat_history) 로 바뀌어 MessagesPlaceholder 자리에 들어간다
  # -> 모델이 이전 대화의 '나는 AI 개발자다.' 를 볼 수는 있다 (history 가 10개를 넘으면 오래된 대화부터 빠진다)
  # -> 다만 시스템 규칙이 "PDF 컨텍스트로만 답하기" 라서, 논문과 무관한 이 질문은 "주어진 정보로는 답변할 수 없습니다." 가 나올 수 있다
'''