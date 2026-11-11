from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('C:\\Users\\manik\\OneDrive\\Desktop\\Govt services Guide\\Everyday Government Services Guide – India.pdf')
document=loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,separators=["\n\n","\n"," ",""])
chunks=splitter.split_documents(document)

from langchain_huggingface import HuggingFaceEmbeddings
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_chroma import Chroma
vectorDB=Chroma.from_documents(
    collection_name="govt_services_rag_bot",
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db",
    collection_configuration={
        "hnsw": {
            "space": "cosine",
            "ef_construction": 200,
            "ef_search": 100,
            "max_neighbors": 32
        }
    }
)

from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(query):
    retrieved_docs = vectorDB.similarity_search(
        query,
        k=10
    )
    pairs = [
        [query, doc.page_content]
        for doc in retrieved_docs
    ]
    scores = reranker.predict(pairs)
    ranked_docs = sorted(
        zip(scores, retrieved_docs),
        key=lambda x: x[0],
        reverse=True
    )
    final_docs = [doc for score, doc in ranked_docs[:5]]
    return final_docs

from langchain_ollama import ChatOllama
llm=ChatOllama(model="llama3.2:3b",temperature=0.5)

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
You are a friendly, intelligent and helpful Government Services Assistant for India.
Your goal is to provide accurate information about Indian government services while maintaining natural, human-like conversations.
Conversation Instructions:
1. If the user greets you (Hello, Hi, Good morning, etc.), respond with a friendly and natural greeting.
2. If the user says thank you or expresses appreciation, respond politely and naturally.
3. If the user says goodbye, bye, see you, or indicates that the conversation is ending, respond with a friendly farewell.
4. If the user engages in casual conversation, respond naturally like a helpful conversational assistant.
5. If the user asks what you can do, briefly explain that you can help with Indian government services and related procedures.
6. Do not force government-service information into greetings, thanks, or casual conversations.
Government Service Instructions:
1. Answer government-service questions using only the provided context.
2. Do not invent facts, procedures, eligibility criteria, fees, deadlines, or URLs.
3. If the required information is unavailable in the context, say:
   "I don't have sufficient information in the provided documents."
4. Explain government procedures in simple, understandable language.
5. Use numbered steps wherever applicable.
6. Include relevant eligibility requirements and required documents only when supported by the context.
7. Mention the source page numbers supporting your answer.
8. Never fabricate source references or page numbers.
Response Style:
- Be friendly, professional, and conversational.
- Use clear and simple English.
- Avoid unnecessarily lengthy responses.
- Avoid repeating the user's question.
- Use headings and bullet points when they improve readability.
- Do not sound robotic or overly formal.
Context:
{context}
User Question:
{question}
Answer:
""")

def format_docs(docs):
    context = []
    for doc in docs:
        context.append(
            f"Source: {doc.metadata.get('source')}\n"
            f"Page: {doc.metadata.get('page', 0) + 1}\n"
            f"Content: {doc.page_content}"
        )
    return "\n\n".join(context)

from langchain_core.output_parsers import StrOutputParser
rag_chain=prompt|llm|StrOutputParser()

def ask_question(query):
    docs = retrieve_and_rerank(query)
    context = format_docs(docs)
    response = rag_chain.invoke({
        "context": context,
        "question": query
    })
    return response,docs

import gradio as gr
def chatbot(message, history):
    docs = retrieve_and_rerank(message)
    context = format_docs(docs)
    response = rag_chain.invoke({
        "context": context,
        "question": message
    })
    return response
interface = gr.ChatInterface(
    fn=chatbot,
    title="🇮🇳 Government Services Assistant",
    description="Your AI assistant for understanding Indian government services.",
    chatbot=gr.Chatbot(
        height=500,
        placeholder="Welcome! Ask me anything about Indian government services."
    ),
    textbox=gr.Textbox(
        placeholder="Ask your question...",
        container=False
    )
)
interface.launch(share=True)