from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

from langchain.chat_models import init_chat_model
# 创建大模型对象

# 方法一： 直接使用模型类
# deepseek_llm = ChatDeepSeek(
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL,
#     model='deepseek-chat'
# )

# 方法二：通过统一方式
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    model_provider='deepseek',
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

qwen_llm = init_chat_model(
    model='qwen-chat',
    model_provider='openai',
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL
)
