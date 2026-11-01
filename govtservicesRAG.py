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
from gtts import gTTS
import tempfile

def govt_services(message, history):
    results=vectorDB.similarity_search(message,k=4)
    context="\n\n".join(doc.page_content for doc in results)
    messages=prompt.invoke({
        "context": context,
        "input": message
    })
    response=llm.invoke(messages)
    answer=response.content
    voice=gTTS(text=answer, lang="en")
    audio_file=tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )
    audio_path=audio_file.name
    audio_file.close()
    voice.save(audio_path)
    return answer, audio_path

import gradio as gr

with gr.Blocks() as demo:
    audio_output=gr.Audio(
        label="LLM Voice Response",
        render=False,
        autoplay=False
    )
    chat=gr.ChatInterface(
        fn=govt_services,
        title="🤖 Govt Services RAG 🤖",
        description="Ask questions about Indian Government Services",
        additional_outputs=[audio_output]
    )
    audio_output.render()
demo.launch(share=True)