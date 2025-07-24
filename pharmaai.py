import streamlit as st
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyD5-FXCZFUt8gn2Chv9tgKiRZ7PlkD5hBg" 

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db



def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(gemini_model_id, GOOGLE_API_KEY):
    llm = ChatGoogleGenerativeAI(
        model=gemini_model_id,
        temperature=0.5
    )
    return llm

def main():
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    st.title("ask PharmaAi!")


    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """

        GEMINI_MODEL_ID="gemini-1.5-flash"
        GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")


        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(gemini_model_id=GEMINI_MODEL_ID, GOOGLE_API_KEY=GOOGLE_API_KEY),  # ✅ added comma here
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )


            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            result_to_show=result+"\nSource Docs:\n"+str(source_documents)
            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ =="__main__":
    main()