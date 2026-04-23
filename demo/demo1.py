"""
基础版-多轮对话
"""

import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from init_llm import deepseek_llm

# 代理服务器
os.environ['http_proxy'] = 'http://127.0.0.1:33210'
os.environ['https_proxy'] = 'http://127.0.0.1:33210'

os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# 指定项目
os.environ['LANGCHAIN_PROJECT'] = 'ChatRobot'

# LangSmith的API Key

# 聊天机器人
# 1. LLM
model = deepseek_llm

# 2. Prompt
prompt = ChatPromptTemplate.from_messages([
    ('system', """
        You are a strict multilingual assistant.
        
        Rules:
        - You MUST respond ONLY in {language}
        - No other languages allowed
        - Do not explain language choice
        """),

    MessagesPlaceholder(variable_name='history'),  # 历史聊天记录插入prompt
    ('human', '{input}')  # 当前用户输入
])

# 3. Parse
parser = StrOutputParser()

# 4. Chain
chain = prompt | model | parser

# 5. 保存聊天的历史记录
store = {}  # k-v -->  sessionID: 历史聊天记录对象


# 接受会话ID，返回历史记录对象
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
c

# chain + memory
do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 去哪里获得历史记录
    input_messages_key='input',  # 哪一段是用户输入
    history_messages_key='history',  # 历史记录放到prompt的哪个变量  --> MessagePlaceholder
)

config1 = {'configurable': {'session_id': 'zs123'}}  # 当前对话sessionID

# 第一轮聊天
resp1 = do_message.invoke(
    input={
        'input': '你好，我是ZhangSan',
        'language': '中文'
    },
    config=config1
)

print(resp1)

# 第二轮聊天
resp2 = do_message.invoke(
    input={
        'input': '请问我的名字是什么？',
        'language': '中文'
    },
    config=config1
)

print(resp2)


# 第三轮聊天   stream
config2 = {'configurable': {'session_id': 'lisi123'}}  # 当前对话sessionID

for resp in do_message.stream(input={'input': '请给我讲一个笑话', 'language': 'English'},
                              config=config2):
    # 每次响应都是一个token
    print(resp, end='-')
