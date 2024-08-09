from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import FewShotChatMessagePromptTemplate
from langchain_teddynote.messages import stream_response
import os

# Ollama 모델을 불러옵니다.
# llm = ChatOllama(model="llama3.1:8b")
# Ollama 모델을 불러옵니다.
# OpenAI API Key
os.environ["OPENAI_API_KEY"] = (
	""
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

examples = [
    {
        "input": '["연애","자취","비밀번호"]',
        "output": '[{"instruct":"연애하는 사이에서, 연인에게 자취방 비밀번호를 알려줘도 되는가?", "A":"알려줘도 된다.", "B":"알려주면 안된다."},{...]'
    },
    {
        "input": '["길고양이","먹이","간식"]',
        "output": '[{"instruct":"길고양이에게 먹이를 줘도 되는가?", "A":"먹이를 줘도 된다.", "B":"먹이를 주면 안된다."},{...]'
    },
    {
        "input": '["의대생","인원수"]',
        "output": '[{"instruct":"전국 의대생 정원 수를 늘리는 것은 타당한가?.", "A":"타당하다.", "B":"타당하지 않다."},{...]'
    },
    {
        "input": '["인싸","애인"]',
        "output": '[{"instruct":"인싸 애인이 나은가 아싸 애인이 나은가?", "A":"인싸 애인이 낫다.", "B":"아싸 애인이 낫다."},{...]'
    },
    {
        "input": '["돈","남자","연애","여자"]',
        "output": '[{"instruct":"데이트 비용은 남자가 더 많이 내야하는가?", "A":"남자가 더 많이 내야한다.", "B":"여자가 더 많이 내야한다."},{...]'
    },
    {
        "input": '["AI","인공지능","일자리"]',
        "output": '[{"instruct":"인공지능의 발전은 일자리 수에 악영향을 미칠 것인가?", "A":"악영향을 미친다.", "B":"좋은 영향을 미친다."},{...]'
    },
    {
        "input": '["백인","흑인","황인"]',
        "output": '[{"instruct":"인종에는 우열이 있는가?", "A":"우열이 있다.", "B":"우열이 없다."},{...]'
    },
]
# 요약을 위한 프롬프트 템플릿 정의
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)

# few shot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# 최종 prompt 
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an ai bot that generates three provocative discussion topics using given words and outputs them in JSON format."),
        few_shot_prompt,
        ("human", "{input}")
    ]
)
# print('----')
# print(final_prompt.format_messages(input='["데이트","비용","남자","여자","비중","비율"]'))
# print('----')
# 요약할 텍스트 정의
stream_response(llm.stream(final_prompt.format_messages(input='["국민대","여자"]')))



"""
chain = prompt | llm | StrOutputParser()
chain.invoke()

stream_response()

# 간결성을 위해 응답은 터미널에 출력됩니다.
answer = chain.invoke(input="")
# 스트리밍 출력
stream_response(answer)
"""