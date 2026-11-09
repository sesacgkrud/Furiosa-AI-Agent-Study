# [목적] 환경 변수 OPENAI_API_KEY 가 등록되어 있는지, 키 일부만 가려서 확인 (Day19)
# [흐름] os.getenv 로 읽기 → 없으면 None → 있으면 길이와 앞 8 글자 · 뒤 4 글자만 출력 (전체 키는 화면에 보이지 않게)
# ※ 7 행 출력 문자열 '키 킬이' 는 '키 길이' 의 오타 (출력 글자만 다르고 동작은 같다)

import os

# os.getenv : 환경 변수 값, 없으면 None (os.environ['키'] 와 달리 오류가 나지 않는다)
key = os.getenv('OPENAI_API_KEY')
if key is None:
    print('OPENAI_API_KEY 없음')
else:
    print('키 킬이 :', len(key))
    # key[:8] 앞 8 글자 + ... + key[-4:] 뒤 4 글자
    print('키 확인 :', key[:8] + "..." + key[-4:])