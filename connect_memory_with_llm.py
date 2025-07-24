import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Shafqat Mahmood/Desktop/pharmaAi/absolute-gantry-466818-m8-eca2a233b069.json"
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# step: 1 setup llm(gemini-1.5-flash-latest with google ai)
GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL_ID = "gemini-1.5-flash"

def load_llm(gemini_model_id):
    llm = ChatGoogleGenerativeAI(
        model=gemini_model_id,
        temperature=0.5
    )
    return llm

# step: 2 connect llm with faiss and creat chain
DB_FAISS_PATH="vectorstore/db_faiss"
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


# Create QA chain for Gemini 1.5 Flash
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(GEMINI_MODEL_ID),  # Uses Gemini model
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
