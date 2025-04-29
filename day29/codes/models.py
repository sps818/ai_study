from dotenv import load_dotenv

load_dotenv()


# 加载阿里大模型
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings


def get_tongyi_chat_model():
    """加载阿里大模型"""
    return ChatTongyi(model="qwen-turbo", temperature=0.1, top_p=0.3, max_tokens=512)


def get_tongyi_embed_model():
    """加载阿里向量化大模型"""
    return DashScopeEmbeddings(model="text-embedding-v3")


if __name__ == "__main__":
    """测试"""
    chat = get_tongyi_chat_model()
    response = chat.invoke("你好")
    print(response)
