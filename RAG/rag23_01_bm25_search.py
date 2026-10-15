# [목적] BM25 키워드 검색 - 질문에 들어 있는 단어가 문서에 그대로 있는지로 문서를 찾는다
#
# [흐름]
#   01. 짧은 문장 8개를 Document 로 만든다 (파일을 불러오지 않고 직접 작성)
#   02. BM25Retriever.from_documents(texts) 로 문서들의 단어 통계를 만든다 (임베딩 · 벡터 DB 가 필요 없다)
#   03. bm25_retriever.k = 3 → 점수가 높은 문서 3개를 가져온다
#   04. invoke(query) 로 검색하고 결과 문장을 출력한다
#
# [BM25 점수]
#   - 질문의 단어가 문서에 많이 나올수록 점수가 높다
#   - 여러 문서에 흔하게 나오는 단어보다 드물게 나오는 단어가 맞을 때 점수가 더 높다
#   - 문서가 길면 점수를 조금 깎는다 (긴 문서가 단어를 많이 가져서 유리해지는 것을 막는다)
#   - 의미(벡터)는 보지 않고 글자가 같은 단어만 본다
#
# [결과 해석]
#   - 기본 설정은 띄어쓰기로만 단어를 나눈다 → 질문 단어 = ['삼성전자의', '반도체', '사업']
#   - '삼성전자는', '반도체를', '반도체는' 처럼 조사가 붙으면 다른 단어가 되어 맞지 않는다
#   - '반도체' 가 조사 없이 그대로 들어 있는 'AI 반도체 기술을' 문장만 점수가 있고 (1.47) 나머지 7개는 0점
#     → 2, 3번째 결과(반도체는 ... / 테슬라는 ...)는 0점끼리 순서대로 채워진 것이라 질문과 관계가 없다
#   - 한국어처럼 조사가 붙는 말은 키워드 검색만으로는 부족하다 → rag23_02 벡터 검색, rag23_03 hybrid 검색으로 비교

from langchain_core.documents import Document
# pip install rank_bm25

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = '삼성전자의 반도체 사업'

#################### BM25 검색 ####################
# BM25는 키워드 기반 검색. Best Match 25

from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(texts)   # 문서 리스트로 BM25 검색기를 만든다
bm25_retriever.k = 3                                    # 검색 결과로 가져올 문서 수

bm25_docs = bm25_retriever.invoke(query)                # Document 리스트로 돌려준다
print('========== BM25 검색 결과 ==========')
for doc in bm25_docs:
    print(doc.page_content)

    # ========== BM25 검색 결과 ==========
    # 삼성전자는 AI 반도체 기술을 연구한다.
    # 반도체는 컴퓨터의 핵심 부품이다.
    # 테슬라는 전기자동차를 제조한다.
