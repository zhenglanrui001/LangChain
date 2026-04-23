from langchain_deepseek import ChatDeepSeek

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# 创建大模型对象

# 方法一： 直接使用模型类
deepseek_llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model='deepseek-chat'
)

resp = deepseek_llm.invoke('Hello')
print(type(resp))
print(resp)