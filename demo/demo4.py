import os
import bs4

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from env_utils import LANGCHAIN_API_KEY
from init_llm import deepseek_llm

# os.environ['http_proxy'] = 'http://127.0.0.1:33210'
# os.environ['https_proxy'] = 'http://127.0.0.1:33210'
#
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_PROJECT'] = 'ChatRobot'
# os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY

os.environ.pop('https_proxy', None)
os.environ.pop('http_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# LLM
llm = deepseek_llm

# 1.加载数据: 网络博客
loader = WebBaseLoader(
    web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent/'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title', 'post-content')),  # 解析3个
    )
)

docs = loader.load()

# 2. 文本切割
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
splits = splitter.split_documents(docs)

# 3. 存储
vector_db = Chroma.from_documents(splits, embedding=DashScopeEmbeddings())

# 4. 检索器
retriever = vector_db.as_retriever()

# 5. 整合

# 创建一个问题的模版
system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. Use three sentences maximum and keep the answer concise.\n

{context}
"""
prompt = ChatPromptTemplate.from_messages(
    [('system', system_prompt),
     MessagesPlaceholder('chat-history'),
     ('human', '{input}')]
)

# 得到chain
chain1 = create_stuff_documents_chain(llm, prompt)  # 问答

# chain2 = create_retrieval_chain(retriever, chain1)  # 检索

# 子链的提示模版
contextualize_q_system_prompt = """Given a chat history and the latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is."""

retriever_history_temp = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat-history'),
        ('human', '{input}')
    ]
)

# 创建一个子链
history_chain = create_history_aware_retriever(llm, retriever, retriever_history_temp)

# 保存问答的历史记录
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]


# 创建父链: 把前面两个链整合
chain = create_retrieval_chain(history_chain, chain1)

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat-history',
    output_messages_key='answer'
)

# 第一轮对话
resp1 = result_chain.invoke(
    {'input': 'What is Task Decomposition?'},
    config={'configurable': {'session_id': 'zhangsan123456'}}
)

print(resp1['answer'])

# 第二轮对话
resp2 = result_chain.invoke(
    {'input': 'What are common ways of doing it?'},
    config={'configurable': {'session_id': 'zhangsan123456'}}
)

print(resp2['answer'])