from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

def get_llm(llm_name):
    if llm_name == "chatgpt-latest":
        return ChatOpenAI()
    elif llm_name == "openai-latest":
        return OpenAI()
    elif llm_name == "openai-latest-1024":
        return OpenAI(max_tokens=1024)
    
    raise ValueError(f"LLM '{llm_name}' not found!")