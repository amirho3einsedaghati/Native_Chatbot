from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os 


load_dotenv()

template = """
You are a helpful Summarizer Chatbot who can just summarize the input text and return.

User : {input_text}

AI : """

prompt = PromptTemplate.from_template(template)

class Summarizer(object):
    def __init__(self) -> None:
        self.llm = ChatNVIDIA(
            model='google/gemma-2-2b-it',
            api_key=os.getenv('NV_API_KEY'),
            max_tokens=128,
            temperature=0.01,
            top_p=.7
        )
        self.chain = prompt | self.llm 


    def summarize(self, mem:list):
        summarized_memory = []
        for item in mem:
            if item['role'].lower() == 'user':
                summarized_memory.append(item)
            else:
                summarized_content = self.chain.invoke(item['content']).content
                summarized_memory.append({'role' : item['role'], 'content' : summarized_content})

        return summarized_memory
