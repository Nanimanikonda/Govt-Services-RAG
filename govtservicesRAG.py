from langchain_community.document_loaders import PyPDFLoader
from gtts import gTTS
import tempfile

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

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

classifier_prompt = ChatPromptTemplate.from_template("""
Classify the user's question into exactly one category:

government:
Questions about Indian government services, schemes, eligibility,
documents, application procedures, fees, portals and government policies.

general:
Questions about all other topics.

Return only: government or general.

Question: {input}
""")
classifier_chain=classifier_prompt|llm|StrOutputParser()

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
chain1=prompt|llm|StrOutputParser()

general_prompt = ChatPromptTemplate.from_template("""
You are a helpful general-purpose AI assistant.
Answer the user's question clearly and accurately using your general knowledge.
If you are uncertain about something, mention that instead of inventing facts.
Question: {input}
Answer:
""")
chain2=general_prompt|llm|StrOutputParser()

def govt_services(message, history):
    intent = classifier_chain.invoke({"input": message}).strip().lower()

    if intent == "government":
        results = vectorDB.similarity_search(message, k=4)

        context = "\n\n".join(
            doc.page_content for doc in results
        )

        answer = chain1.invoke({
            "context": context,
            "input": message
        })

    else:
        answer = chain2.invoke({"input": message})

    try:
        audio_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        ).name

        tts = gTTS(text=answer, lang="en")
        tts.save(audio_path)

        return answer, audio_path

    except Exception as e:
        print("Text-to-speech error:", e)
        return answer, None


import gradio as gr

with gr.Blocks() as demo:

    audio_output = gr.Audio(
        label="Listen to the response",
        type="filepath",
        autoplay=True
    )

    chat = gr.ChatInterface(
        fn=govt_services,
        additional_outputs=[audio_output],
        title="🤖 Govt Services RAG 🤖",
        description="Ask questions about Indian Government Services"
    )

demo.launch(share=True)