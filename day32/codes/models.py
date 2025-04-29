from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatTongyi

def get_chat():
    return ChatTongyi(model="qwen-turbo", temperature=0.1)