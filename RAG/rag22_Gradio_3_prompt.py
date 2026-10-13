# rag22_Gradio_2_Similarity.py 베이스
# [변경] 프롬프트 규칙을 바꿔서 인사 · 이전 대화 질문에는 자연스럽게 답하고, 논문 질문은 컨텍스트로만 답한다

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

# [변경] score_threshold 0.7 -> 0.2 (rag20 에서 Qwen3 기준으로 정했던 값)
#  0.7 이면 논문 질문도 검색 결과가 0개라 답할 수 없다
#  0.2 면 논문 질문(0.42 ~ 0.59) 만 문서가 들어오고, "안녕"(0.11) · "삼성전자"(-0.03) 는 0개가 된다
retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 2, 'score_threshold': 0.2},
)

#06. 답변 모델
model = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

#07. 프롬프트
prompt = ChatPromptTemplate.from_messages([
    # [변경] 규칙 1, 2 추가 : 검색 문서가 없는 인사 · 이전 대화 질문도 답할 수 있게 한다
    #  rag22_1 · rag22_2 규칙은 "PDF 만 보고 답하기" 라서 "안녕", "내가 누구게" 도 "주어진 정보로는 답변할 수 없습니다." 가 나왔다
    ('system', '''너는 Attention Is All You Need.pdf 를 보고 답하는 챗봇이다. 아래 규칙을 반드시 지켜라.
1. "안녕", "고마워" 같은 인사나 일상 대화는 자연스럽게 답해라.
2. 사용자가 자기 자신에 대해 묻거나("내가 누구게" 등) 이전 대화에서 알려 준 정보(이름, 직업 등) 를 물으면, 컨텍스트와 상관없이 이전 대화를 보고 답해라.
3. 그 밖의 질문은 컨텍스트에 쓰여 있는 내용으로만 답해라.
   컨텍스트에 없는 정의, 배경 설명, 숫자, 인물, 예시를 네가 아는 지식이나 일반 상식으로 덧붙이지 마라.
4. 3번 질문인데 컨텍스트에 관련 내용이 없으면, 다른 말 없이 정확히 "{no_answer}" 한 문장만 출력해라.

이전 대화 :
{history}

컨텍스트 :
{context}'''),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}'),
])

prompt = prompt.partial(
    no_answer=NO_ANSWER,
    history='(컨텍스트 아래에 이어지는 대화 메시지 참고)',
)

#08. 체인 만들기
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever

docu_chain = create_stuff_documents_chain(model, prompt)

rephrase_prompt = ChatPromptTemplate.from_messages([
    ('system', '이전 대화를 참고해서, 마지막 질문을 이전 대화 없이도 이해할 수 있는 하나의 질문으로 다시 써라. 답하지 말고 질문만 출력해라.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}'),
])
history_retriever = create_history_aware_retriever(model, retriever, rephrase_prompt)

rag_chain = create_retrieval_chain(history_retriever, docu_chain)

#10. 챗봇 함수
def answer_invoke(message, history):
    chat_history = []

    for item in history[-10:]:
        content = item['content']

        if item['role'] == 'user':
            chat_history.append(HumanMessage(content=content))
        elif item['role'] == 'assistant':
            chat_history.append(AIMessage(content=content))

    response = rag_chain.invoke({
        'input' : message,
        'chat_history' : chat_history,

    })

    for i, doc in enumerate(response['context'], start=1):
        print(f'\n검색 문서 {i}')
        print('출처: ', doc.metadata.get('source'))
        print('내용: ', doc.page_content)
        print('========================================')

    # [삭제] if not response['context']: return NO_ANSWER
    #  "안녕" 처럼 검색 문서가 0개인 질문도 모델이 답해야 하므로 지우고, 거절은 프롬프트 규칙 4번에 맡긴다

    return response['answer']

#11. Gradio 화면
demo = gr.ChatInterface(
    fn=answer_invoke,
    title='📊 Attention Is All You Need RAG 챗봇',
    # [변경] 인사 · 이전 대화 질문에도 답한다는 설명으로 바꿈
    description='Qwen3 임베딩으로 만든 FAISS 벡터 DB 에서 Attention Is All You Need 논문을 보고 답합니다. 인사나 이전 대화에서 알려 준 내용에는 자연스럽게 답하고, 논문에 없는 내용은 답하지 않습니다.',
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
