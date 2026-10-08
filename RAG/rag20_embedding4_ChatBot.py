import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings     # [변경] OpenAIEmbeddings 대신 로컬 HuggingFace 임베딩 사용

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
# [변경] OpenAI API 대신 Qwen3-Embedding-0.6B 를 내 PC(CPU)에서 직접 돌린다 -> 임베딩 API 비용 없음
#  벡터 차원 1024 (OpenAI small 은 1536), 출력 벡터 길이가 1 로 정규화되어 나온다
#  처음 실행할 때 모델(약 1.2GB)을 내려받아 ~/.cache/huggingface 에 저장하고, 이후에는 저장된 모델을 쓴다
embeddings = HuggingFaceEmbeddings(
    model_name='Qwen/Qwen3-Embedding-0.6B',
    model_kwargs={'device': 'cpu'},
)

#05. FAISS
# [변경] faiss.IndexFlatL2 로 따로 만들던 인덱스는 쓰이지 않아 삭제 (FAISS.from_documents 가 인덱스를 직접 만든다)
db = FAISS.from_documents(
    documents=split_doc,
    embedding=embeddings,
)

# [변경] Faiss19 -> Faiss20 : rag19 (OpenAI 1536 차원) DB 를 Qwen3 (1024 차원) DB 로 덮어쓰지 않게 경로를 분리
DB_PATH = './_db/Faiss20'
db.save_local(
    folder_path=DB_PATH,
    index_name='transformer_index20'
)

db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='transformer_index20',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# [변경] score_threshold 0.05 -> 0.2 : Qwen3 기준으로 다시 맞춘 값
#  논문 질문은 0.42 ~ 0.59, "삼성전자의 사업 전망은?" 은 -0.03, "오늘 점심 메뉴 추천해줘" 는 0.09
#  0.05 로 두면 점심 메뉴 같은 무관한 질문도 통과해 모델을 부르게 된다
retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 4, 'score_threshold': 0.2},
)

#06. 답변 모델
model = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

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
    ('human', '{input}'),
])

#08. 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)

#09. Gradio 대화 기록을 문자열로 바꾸기
def content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join(c.get('text', '') for c in content if isinstance(c, dict))
    return str(content)

def history_to_text(history, max_turns=3):
    lines = []
    for h in history[-max_turns * 2:]:
        who = '사용자' if h['role'] == 'user' else '챗봇'
        lines.append(f'{who}: {content_to_text(h["content"])}')
    return '\n'.join(lines) if lines else '(없음)'

def last_user_question(history):
    for h in reversed(history):
        if h['role'] == 'user':
            return content_to_text(h['content'])
    return ''

#10. 챗봇 함수
def answer_invoke(message, history):
    search_query = f'{last_user_question(history)} {message}'.strip()
    docs = retriever.invoke(search_query)

    if not docs:
        return NO_ANSWER

    answer = docu_chain.invoke({
        'context': docs,
        'input': message,
        'history': history_to_text(history),
        'no_answer': NO_ANSWER,
    })

    if len(answer) < 120 and any(word in answer for word in REFUSAL_WORDS):
        return NO_ANSWER

    sources = sorted({os.path.basename(d.metadata['source']) for d in docs})
    return f'{answer}\n\n📄 출처: {", ".join(sources)}'

#11. Gradio 화면
demo = gr.ChatInterface(
    fn=answer_invoke,
    title='📊 Attention Is All You Need RAG 챗봇',
    # [변경] 임베딩 모델이 바뀐 것을 화면 설명에도 표시
    description='Qwen3 임베딩으로 만든 FAISS 벡터 DB 에서 Attention Is All You Need 논문만 보고 답합니다. 문서에 없는 내용은 답하지 않고, 답변 아래에 출처 파일을 보여줍니다.',
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
