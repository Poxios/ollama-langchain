from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import FewShotChatMessagePromptTemplate
from langchain_teddynote.messages import stream_response


# Ollama 모델을 불러옵니다.
llm = ChatOllama(model="eeve:latest")

examples = [
    {
        "input": '"자동차"',
        "output": '"car"'
    },
    {
        "input": '"길고양이에게 먹이를 줘도 되는가?"',
        "output": '"Is it okay to feed stray cats?"'
    }
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
        ("system", "You are an AI bot that translates Korean to English."),
        few_shot_prompt,
        ("human", "{input}")
    ]
)

# 요약할 텍스트 정의
input_instruct='"전국 의대생의 정원을 늘리는 것이 타당한가?"'
stream_response(llm.stream(final_prompt.format_messages(input=input_instruct)))



"""
chain = prompt | llm | StrOutputParser()
chain.invoke()

stream_response()

# 간결성을 위해 응답은 터미널에 출력됩니다.
answer = chain.invoke(input="")
# 스트리밍 출력
stream_response(answer)
"""