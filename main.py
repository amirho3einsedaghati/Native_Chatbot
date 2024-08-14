import streamlit as st
from tools.chatbot import QuestionAnswering



models = {
        'llama-3.1-405b-instruct' : 'meta/llama-3.1-405b-instruct',
        'gemma-2-27b-it' : 'google/gemma-2-27b-it',
        'mistral-nemo-12b-instruct' : 'nv-mistralai/mistral-nemo-12b-instruct',
        'nemotron-4-340b-instruct' : 'nvidia/nemotron-4-340b-instruct',
        'phi-3-medium-128k-instruct' : 'microsoft/phi-3-medium-128k-instruct',
        'arctic' : 'snowflake/arctic'
}

def main():  
    st.header("🤖 Your Native Chatbot is ready to help")
    st.markdown("**It helps you write and talk like a native speaker. So, What are you waiting for ? Let's go 😀**")

    model_key = st.sidebar.selectbox('Powered by', list(models.keys()))
    model_value = models[model_key]
    memory = []
    chatbot = QuestionAnswering(model_name=model_value, memory=memory)

    curr_question = st.text_area(
        'enter your prompt',
        placeholder="Talk to your Native Chatbot!",
        label_visibility="hidden"
    )

    if st.button("Generate Answer"):
        try:
            chatbot.generate_answer(curr_question)
        except:
            st.warning(body="Refresh the page or Try it again later.", icon="🤖")

main()

