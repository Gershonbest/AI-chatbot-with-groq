import streamlit as st
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os 

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

# session state variable
if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[]



st.set_page_config(page_title="Personal ChatBot with Groq AI", page_icon='🤖')
st.title("Personal conversational chatbot:iphone:")
st.write("Hello! How can I help you today. I am here to help you answer your questions and have a good time. Select a prefered language model from the side bar.")

# Add customization options to the sidebar
st.sidebar.title('Select an LLM')
model = st.sidebar.selectbox(
    'Choose a model',
    ['mixtral-8x7b-32768', 'llama2-70b-4096']
)
conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 3)

memory=ConversationBufferWindowMemory(k=conversational_memory_length)

user_query = st.chat_input("Type your prompt here:")

# Initialize Groq Langchain chat object and conversation
groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
)

def get_response(query, llm, chat_history):
    template = """
     Your Name is Andy and you are a helpful assistant. Answer the following questions considering the history of the conversation conversation:
    
     Chat history: {chat_history}

     User question: {user_question}
    """

    prompt  = PromptTemplate(input_variables=['chat_history', 'user_question'], template= template)
    chain  = LLMChain(llm=llm, prompt=prompt)

    output = chain.invoke(query, chat_history)
    return output


conversation = ConversationChain(
            llm=groq_chat,
            memory=memory
    )

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="🧑‍💻"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI", avatar="🤖"):
            st.markdown(message.content)

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message('Human', avatar="🧑‍💻"):
        st.markdown(user_query)

    with st.chat_message("AI", avatar="🤖"):
        ai_response = conversation(user_query)
        st.markdown(ai_response['response'])

    st.session_state.chat_history.append(AIMessage(ai_response['response']))


 
