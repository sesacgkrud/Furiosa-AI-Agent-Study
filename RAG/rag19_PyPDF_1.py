# [목적] PyPDFLoader 로 PDF 파일(Attention Is All You Need 논문) 을 불러와 어떤 형태로 들어오는지 확인한다
#   - TextLoader 가 텍스트 파일 1개를 Document 1개로 불러왔다면, PyPDFLoader 는 PDF 를 페이지마다 Document 1개로 불러온다
#   - load() 결과 = Document 리스트 -> 15페이지라 len 15
#   - Document = page_content (그 페이지의 글자) + metadata (source 파일 경로, page 번호(0부터), page_label, total_pages 등)
#   - 다음 파일(rag19_PyPDF_2_ChatBot) 에서 이 Document 들을 청크로 잘라 FAISS 에 저장한다

from langchain_community.document_loaders import TextLoader, PyPDFLoader

path = './_data/'
pdf_loader = PyPDFLoader(path + 'Attention Is All You Need.pdf')
pdf_docs = pdf_loader.load()

print(type(pdf_docs)) # <class 'list'>
print(len(pdf_docs))  # 15

# print(pdf_docs)
# [Document(metadata={...},
#           page_content='Provided proper attribution is provided, Google hereby grants ...'
# )]

print('====================')
print(pdf_docs[0])                     # 첫 페이지 (page 0) : 제목 · 저자 · Abstract
print('====================')

# page_content='Provided proper attribution is provided, Google hereby grants permission to
# reproduce the tables and figures in this paper solely for use in journalistic or
# scholarly works.
# Attention Is All You Need
# Ashish Vaswani∗
# Google Brain
# avaswani@google.com
# Noam Shazeer∗
# Google Brain
# noam@google.com
# Niki Parmar∗
# Google Research
# nikip@google.com
# Jakob Uszkoreit∗
# Google Research
# usz@google.com
# Llion Jones∗
# Google Research
# llion@google.com
# Aidan N. Gomez∗†
# University of Toronto
# aidan@cs.toronto.edu
# Łukasz Kaiser∗
# Google Brain
# lukaszkaiser@google.com
# Illia Polosukhin∗‡
# illia.polosukhin@gmail.com
# Abstract
# The dominant sequence transduction models are based on complex recurrent or
# convolutional neural networks that include an encoder and a decoder. The best
# performing models also connect the encoder and decoder through an attention
# mechanism. We propose a new simple network architecture, the Transformer,
# based solely on attention mechanisms, dispensing with recurrence and convolutions
# entirely. Experiments on two machine translation tasks show these models to
# be superior in quality while being more parallelizable and requiring significantly
# less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-
# to-German translation task, improving over the existing best results, including
# ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
# our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
# training for 3.5 days on eight GPUs, a small fraction of the training costs of the
# best models from the literature. We show that the Transformer generalizes well to
# other tasks by applying it successfully to English constituency parsing both with
# large and limited training data.
# ∗Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started
# the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and
# has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head
# attention and the parameter-free position representation and became the other person involved in nearly every
# detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and
# tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and
# efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and
# implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating
# our research.
# †Work performed while at Google Brain.
# ‡Work performed while at Google Research.
# 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
# arXiv:1706.03762v7  [cs.CL]  2 Aug 2023' metadata={'producer': 'pdfTeX-1.40.25', 'creator': 'LaTeX with hyperref', 'creationdate': '2024-04-10T21:11:43+00:00', 'author': '', 'keywords': '', 'moddate': '2024-04-10T21:11:43+00:00', 'ptex.fullbanner': 'This is pdfTeX, Version 3.141592653-2.6-1.40.25 (TeX Live 2023) kpathsea version 6.3.5', 'subject': '', 'title': '', 'trapped': '/False', 'source': './_data/Attention Is All You Need.pdf', 'total_pages': 15, 'page': 0, 'page_label': '1'}