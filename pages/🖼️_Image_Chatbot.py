import time
import os
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import datetime
from PIL import Image
import easyocr
import numpy as np

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
api_key = GOOGLE_API_KEY
new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

# Ensure data folder exists
os.makedirs('data/', exist_ok=True)

image_link = []

try:
    past_chats = joblib.load('data/past_chats_list')
except:
    past_chats = {}

def get_file_text(files):
    text = ""
    reader = easyocr.Reader(['en'])
    if files:
        for file in files:
            image = Image.open(file)
            st.image(image)
            text += f"\n--- Text from {file.name} ---\n"
            results = reader.readtext(np.array(image))
            text += " ".join([i[1] for i in results])
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question in a detailed and expressive manner, providing as much relevant information from the provided context as possible. Make sure the response is storytelling and engaging. 

    If the context is not provided or the context is empty:
    Answer like a normal AI chatbot with a storytelling effect.
    
    If the answer is not in the provided context:
    Answer like a normal chatbot if there is no context provided and specify that the answer is not in the provided context.  
    
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

@st.dialog("Clear chat history?")
def modal():
    button_cols = st.columns([1, 1])
    if button_cols[0].button("Yes"):
        clear_chat_history()
        st.rerun()
    elif button_cols[1].button("No"):
        st.rerun()

def clear_chat_history():
    st.session_state.pop('chat_id', None)
    st.session_state.pop('messages', None)
    st.session_state.pop('gemini_history', None)
    for file in Path('data/').glob('*'):
        file.unlink()

# Sidebar
with st.sidebar:
    st.write('# Sidebar Menu')
    chat_options = [new_chat_id] + list(past_chats.keys())
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = st.selectbox('Pick a past chat', options=chat_options, format_func=lambda x: past_chats.get(x, 'New Chat'))
    else:
        st.session_state.chat_id = st.selectbox('Pick a past chat', options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()), index=1, format_func=lambda x: past_chats.get(x, 'New Chat'))

    if st.button("Clear Chat History", key="clear_chat_button"):
        modal()

    image_files = st.file_uploader("Upload Image Files", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    if st.button("Submit & Process", key="process_button", disabled=not image_files):
        with st.spinner("Processing..."):
            raw_text = get_file_text(image_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)

    st.session_state.chat_title = f'Image-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'

st.write('# Chat with Gemini')

# Chat history management
try:
    st.session_state.messages = joblib.load(f'data/{st.session_state.chat_id}-st_messages')
    st.session_state.gemini_history = joblib.load(f'data/{st.session_state.chat_id}-gemini_messages')
except:
    st.session_state.messages = []
    st.session_state.gemini_history = []

st.session_state.model = genai.GenerativeModel('gemini-1.5-pro-latest')
st.session_state.chat = st.session_state.model.start_chat(history=st.session_state.gemini_history)

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [{"role": "✨", "content": "Hey there, I'm your Text Extraction chatbot. Please upload files in the sidebar for more context."}]

for message in st.session_state.messages:
    with st.chat_message(name=message.get('role', 'user'), avatar=message.get('avatar', None)):
        st.markdown(message['content'])

if prompt := st.chat_input('Your message here...'):
    if st.session_state.chat_id not in past_chats:
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')

    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({"role": 'user', "content": prompt})

    with st.spinner("Waiting for AI response..."):
        response = user_input(prompt)

    with st.chat_message(name=MODEL_ROLE, avatar=AI_AVATAR_ICON):
        st.markdown(response)

    st.session_state.messages.append({"role": MODEL_ROLE, "content": response, "avatar": AI_AVATAR_ICON})
    st.session_state.gemini_history = st.session_state.chat.history

    joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
    joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')
