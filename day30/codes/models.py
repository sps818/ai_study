from dotenv import load_dotenv
load_dotenv()
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi

def get_embed():
    """
        连接模型
    """
    return DashScopeEmbeddings(model="text-embedding-v3")

def get_chat():
    """
        连接模型
    """
    return ChatTongyi(model="qwen-turbo", temperature=0.1, top_p=0.7)