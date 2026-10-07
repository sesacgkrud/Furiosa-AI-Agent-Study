import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import gradio as gr

from dotenv import load_dotenv
load_dotenv()                                                       # .env 파일의 값을 환경 변수로 읽어온다

api_key = os.environ['MONOROUTER_API_KEY'].strip()                  # monorouter API 키 (앞뒤 공백 · 줄바꿈 제거)
base_url = 'https://monogpt.kr/api/monorouter/v1'                   # OpenAI 대신 monorouter 주소로 요청을 보낸다

NO_ANSWER = '주어진 정보로는 답변할 수 없습니다.'                     # 문서에 없는 질문일 때 돌려줄 고정 문구
REFUSAL_WORDS = ['수 없', '있지 않', '되지 않']                    # 모델이 거절할 때 쓰는 표현들 (답변할 수 없 / 포함되어 있지 않 / 제공되지 않 등)

#01. 임베딩 모델
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',                                 # 저장할 때와 같은 임베딩 모델을 써야 검색이 맞는다
    api_key=api_key,
    base_url=base_url,
)

#02. 저장해 둔 FAISS 벡터 DB 불러오기 (rag17_FAISS_1_save.py 에서 저장)
DB_PATH = './_db/Faiss17'

db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,                           # 내가 직접 만든 pkl 파일이라 불러오기를 허용한다
)

print(f'FAISS 에 저장된 문서 수: {db.index.ntotal}')                 # 18 (samsung 9 + nvidia 9)

#03. 검색기 - [고도화 1] 관련 없는 질문 거르기
#  similarity_score_threshold : 유사도 점수가 기준보다 낮은 문서는 버린다 -> DB 에 없는 질문이면 빈 리스트가 나온다
#  k=4 : 제목 한 줄짜리 청크("엔비디아 사업 전망")가 1등으로 뽑히는 경우가 있어서 2개로는 내용이 부족하다 -> 4개까지 가져온다
#  기준값 0.05 는 실제 점수를 찍어 보고 정했다
#    DB 안 질문 (삼성 사업, 엔비디아 위험 요인 등)  : 0.139 ~ 0.493
#    DB 밖 질문 (김치찌개, 대한민국 수도, 날씨 등)  : -0.125 이하
retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 4, 'score_threshold': 0.05},
)

#04. 답변 모델
model = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

#05. 프롬프트 - [고도화 1] 컨텍스트 밖 지식 금지 + [고도화 2] 이전 대화 {history} 추가
#  from_template 한 덩어리로 쓰면 규칙이 질문과 섞여 약하게 먹힌다
#  -> from_messages 로 규칙은 system 메시지, 질문은 human 메시지로 나눠서 규칙을 더 강하게 지키게 한다
prompt = ChatPromptTemplate.from_messages([
    ('system', '''너는 삼성전자 · 엔비디아 전망 문서만 보고 답하는 챗봇이다. 아래 규칙을 반드시 지켜라.
1. 컨텍스트에 질문과 관련된 내용이 있으면, 그 내용을 정리해서 답해라. 질문에 딱 맞는 문장이 없어도 관련 내용이 있으면 답한다.
2. 답변의 모든 문장은 컨텍스트에 쓰여 있는 내용으로 확인할 수 있어야 한다.
   컨텍스트에 없는 정의, 배경 설명, 숫자, 인물, 예시를 네가 아는 지식이나 일반 상식으로 덧붙이지 마라.
3. 컨텍스트에 질문과 관련된 내용이 전혀 없을 때만, 다른 말 없이 정확히 "{no_answer}" 한 문장만 출력해라.
4. 이전 대화는 "그 회사", "거기" 같은 말이 무엇을 가리키는지 파악하는 데에만 사용해라.

이전 대화 :
{history}

컨텍스트 :
{context}'''),
    ('human', '{input}'),
])

#06. 체인 만들기
#  rag15 · rag16 의 create_retrieval_chain 은 검색 결과가 비어도 모델을 부른다
#  -> 검색을 따로 해서 비어 있으면 모델을 부르지 않고, 문서가 있을 때만 docu_chain 을 실행한다
docu_chain = create_stuff_documents_chain(model, prompt)            # 찾은 문서를 {context} 에 채워 모델에 넘긴다

#07. [고도화 2] Gradio 대화 기록을 문자열로 바꾸기
#  Gradio 6 의 history : [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}, ...]
#  content 가 문자열이 아니라 [{'type': 'text', 'text': ...}] 리스트로 올 때도 있어서 둘 다 처리한다
def content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join(c.get('text', '') for c in content if isinstance(c, dict))
    return str(content)

def history_to_text(history, max_turns=3):                          # 최근 3턴(질문+답변 6개)만 넣어 프롬프트가 너무 길어지지 않게 한다
    lines = []
    for h in history[-max_turns * 2:]:
        who = '사용자' if h['role'] == 'user' else '챗봇'
        lines.append(f'{who}: {content_to_text(h["content"])}')
    return '\n'.join(lines) if lines else '(없음)'

def last_user_question(history):                                    # 직전 사용자 질문 (없으면 빈 문자열)
    for h in reversed(history):
        if h['role'] == 'user':
            return content_to_text(h['content'])
    return ''

#08. 챗봇 함수
def answer_invoke(message, history):
    # [고도화 2] 검색어 = 직전 질문 + 현재 질문
    #  "그 회사의 위험 요인은?" 만으로는 어느 회사인지 몰라 점수가 0.009 로 걸러진다 -> 앞 질문을 붙여서 검색한다
    search_query = f'{last_user_question(history)} {message}'.strip()
    docs = retriever.invoke(search_query)

    # [고도화 1] 걸러지고 남은 문서가 없으면 모델을 부르지 않고 바로 고정 문구 반환 (API 비용 절약)
    if not docs:
        return NO_ANSWER

    answer = docu_chain.invoke({
        'context': docs,
        'input': message,
        'history': history_to_text(history),
        'no_answer': NO_ANSWER,
    })

    # [고도화 1] 모델이 "답할 수 없다"를 다른 말로 표현해도 고정 문구로 통일하고 출처를 붙이지 않는다
    #  ("명시되어 있지 않습니다", "답변드릴 수 없습니다" 처럼 매번 표현이 달라서 문구 하나로는 못 잡는다)
    #  길이 120자 미만일 때만 거절로 본다 -> 답을 충분히 한 뒤 "다만 ~은 포함되어 있지 않다" 를 덧붙인 답변까지 지우지 않기 위해
    if len(answer) < 120 and any(word in answer for word in REFUSAL_WORDS):
        return NO_ANSWER

    # [고도화 3] 출처 표시 : metadata['source'] 경로에서 파일명만 뽑아 중복 없이 붙인다
    sources = sorted({os.path.basename(d.metadata['source']) for d in docs})
    return f'{answer}\n\n📄 출처: {", ".join(sources)}'

#09. [고도화 4] Gradio 화면
#  examples 버튼과 Chatbot placeholder 는 대화가 비어 있을 때 첫 화면에 보인다 (대화를 시작하면 사라진다)
demo = gr.ChatInterface(
    fn=answer_invoke,
    title='📊 삼성전자 · 엔비디아 RAG 챗봇',
    description='FAISS 벡터 DB 에 저장된 사업 전망 문서만 보고 답합니다. 문서에 없는 내용은 답하지 않고, 답변 아래에 출처 파일을 보여줍니다.',
    chatbot=gr.Chatbot(
        height=500,                                                 # 대화창 높이
        placeholder='### 무엇이든 물어보세요\n아래 예시 질문을 눌러 시작할 수 있습니다.',   # 대화가 없을 때 대화창 가운데에 보이는 안내 문구
    ),
    textbox=gr.Textbox(placeholder='삼성전자나 엔비디아의 사업 전망을 물어보세요', scale=7),   # 입력창 안내 문구
    examples=[
        '삼성전자의 주요 사업은 뭐야?',
        '엔비디아의 데이터센터 사업은 어때?',
        '그 회사의 위험 요인은?',                                      # 바로 앞 질문을 이어받는 후속 질문 예시
        '김치찌개 맛있게 끓이는 법 알려줘',                             # 문서에 없는 질문 -> 걸러지는지 확인용
    ],
)

demo.launch(theme=gr.themes.Soft())                                 # Gradio 6 부터 theme 는 launch() 에서 지정한다 (Soft : 둥글고 부드러운 테마)
