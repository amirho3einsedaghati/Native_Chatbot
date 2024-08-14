from langchain_nvidia_ai_endpoints import ChatNVIDIA
import streamlit as st
from tools.summarizer import Summarizer
from dotenv import load_dotenv
import os 



load_dotenv()

class QuestionAnswering(object):
    def __init__(self, model_name, memory) -> None:
        self.model_name = model_name
        self.memory = memory
        self.llm = ChatNVIDIA(
            model=self.model_name,
            api_key=os.getenv('NV_API_KEY'),
            max_tokens=700,
            temperature=0.01,
            top_p=.7
        ) 
        self.summarizer = Summarizer()


    def generate_answer(self, question:str): 
        if self.model_name in ['google/gemma-2-27b-it', 'microsoft/phi-3-medium-128k-instruct']:
            prompt = [
                {
                    'role' : "assistant",
                    'content' : "You are a helpful and native chatbot who can guide clients to write and talk naturally\
                                and reduce their mistakes in the different aspects of their English skills. And please\
                                guide users in short and concise answers."
                },
            ]
        
        else:
            prompt = [
                {
                    'role' : "system",
                    'content' : "You are a helpful and native chatbot who can guide clients to write and talk naturally\
                                and reduce their mistakes in the different aspects of their English skills. And please\
                                guide users in short and concise answers."
                },
            ]

        with st.expander('Question'):
            st.write('User: ', question)

        if len(self.memory) != 0:
            # Save the last 2 conversations
            short_term_memory = self.memory[-4:]
            try:
                summarized_short_term_memory = self.summarizer.summarize(short_term_memory)
            except:
                st.warning(body="Refresh the page or Try it again later.", icon="🤖")
            else:
                prompt.extend(summarized_short_term_memory)
        
        user_dict = {'role' : 'user', 'content' : question}
        self.memory.append(user_dict)
        prompt.append(user_dict)

        res = self.llm.invoke(prompt)

        assistant_dict = {'role' : res.response_metadata['role'], 'content' : res.content}
        self.memory.append(assistant_dict)

        with st.expander('Answer'):
            st.write("Assistant: ", assistant_dict['content'])
    

