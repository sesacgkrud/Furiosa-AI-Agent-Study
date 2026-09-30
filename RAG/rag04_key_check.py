import os

key = os.getenv('OPENAI_API_KEY')
if key is None:
    print('OPENAI_API_KEY 없음')
else:
    print('키 킬이 :', len(key))
    print('키 확인 :', key[:8] + "..." + key[-4:])