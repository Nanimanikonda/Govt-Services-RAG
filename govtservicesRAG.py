from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('C:\\Users\\manik\\OneDrive\\Desktop\\Govt services Guide\\Everyday Government Services Guide – India.pdf')
document=loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=200,separators=["\n\n", "\n", " ", ""])
chunks=text_splitter.split_documents(document)

from langchain_huggingface import HuggingFaceEmbeddings

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_chroma import Chroma

vectorDB=Chroma.from_documents(
    documents=chunks,
    collection_name='EverydayGovtServices',
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

from langchain_ollama import ChatOllama

llm=ChatOllama(model="llama3.2:3b",temperature=0)

from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template("""
You are an expert, empathetic assistant specialized in Indian Government Services.
Answer the user's question accurately using ONLY the provided context below.

### Guidelines:
1. Extract step-by-step procedures, document checklists, official portal links, and fee structures directly from the context.
2. Structure your response clearly using markdown bullet points, numbered lists, or tables where appropriate.
3. If the answer or specific detail (like exact fee or portal URL) cannot be found or verified in the context, explicitly state: "Information not available in the official guide." Do not guess or fabricate information.
4. Keep the tone helpful, clear, and direct.

Context:
{context}

Question: 
{input}

Answer:
""")

def govt_services(message, history):
    question = message.strip().lower()
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    thanks = ["thanks", "thank you", "ok thank you", "okay thank you", "thanks a lot"]
    endings = ["bye", "goodbye", "see you", "talk to you later"]
    if question in greetings:
        return "Hello! How can I help you with Indian Government Services today? (e.g., Passports, PAN, Aadhaar, Driving Licences, Voter IDs, EPFO, or AP State Certificates)"
    if question in thanks:
        return "You're welcome! Feel free to ask if you have any more questions about government forms, fees, or procedures."
    if question in endings:
        return "Goodbye! Have a great day."
    results = vectorDB.similarity_search(message, k=4)
    context = "\n\n".join(doc.page_content for doc in results)
    messages = prompt.invoke({"context": context, "input": message})
    response = llm.invoke(messages)
    return response.content

import gradio as gr

demo=gr.ChatInterface(
    fn=govt_services,
    title="🤖 Govt Services RAG 🤖",
    description="Got stuck with Govt. Services! I Can Help You"
)
demo.launch(share=True)