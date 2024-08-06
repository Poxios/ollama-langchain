from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import FewShotChatMessagePromptTemplate
from langchain_teddynote.messages import stream_response


# Ollama 모델을 불러옵니다.
llm = ChatOllama(model="llama3.1:8b")

examples = [
    {
        "input": '["relationship", "apartment", "password"]',
        "output": '{"instruct":"Is it okay to share the password of your apartment with your partner?", "A":"It is okay.", "B":"It is not okay."}'
    },
    {
        "input": '["stray cat", "food", "snack"]',
        "output": '{"instruct":"Is it okay to feed stray cats?", "A":"It is okay.", "B":"It is not okay."}'
    },
    {
        "input": '["medical student", "number"]',
        "output": '{"instruct":"Is it reasonable to increase the number of medical students nationwide?", "A":"It is reasonable.", "B":"It is not reasonable."}'
    },
    {
        "input": '["extrovert", "partner"]',
        "output": '{"instruct":"Is it better to have an outgoing partner or a reserved partner?", "A":"An outgoing partner is better.", "B":"A reserved partner is better."}'
    },
    {
        "input": '["money", "man", "relationship", "woman"]',
        "output": '{"instruct":"Should men pay more for dates?", "A":"Men should pay more.", "B":"Women should pay more."}'
    },
    {
        "input": '["AI", "artificial intelligence", "jobs"]',
        "output": '{"instruct":"Will the development of artificial intelligence negatively impact the number of jobs?", "A":"It will have a negative impact.", "B":"It will have a positive impact."}'
    },
    {
        "input": '["white", "black", "asian"]',
        "output": '{"instruct":"Is there a hierarchy among races?", "A":"There is a hierarchy.", "B":"There is no hierarchy."}'
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
        ("system", "You are an AI bot that generates two opposing arguments in JSON format with a provocative discussion topic given the words."),
        few_shot_prompt,
        ("human", "{input}")
    ]
)
# print('----')
# print(final_prompt.format_messages(input='["데이트","비용","남자","여자","비중","비율"]'))
# print('----')
# 요약할 텍스트 정의
stream_response(llm.stream(final_prompt.format_messages(input='["movie", "eunwoo-cha"]')))



"""
chain = prompt | llm | StrOutputParser()
chain.invoke()

stream_response()

# 간결성을 위해 응답은 터미널에 출력됩니다.
answer = chain.invoke(input="")
# 스트리밍 출력
stream_response(answer)
"""