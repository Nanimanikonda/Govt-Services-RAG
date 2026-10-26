# Government Services RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that helps users find information about government schemes, public services, eligibility criteria, required documents, and application procedures using official government information.

The system retrieves relevant information from government documents and uses a Large Language Model (LLM) to generate context-based answers, helping users understand public services more easily.

## Features

- **Government Scheme Search:** Find information about government schemes and welfare programs.
- **Eligibility Information:** Understand eligibility criteria based on retrieved documents.
- **Document Requirements:** Learn which documents are required for applications.
- **Application Guidance:** Get step-by-step information about application procedures.
- **Semantic Search:** Retrieve relevant information using natural-language questions.
- **Source-Based Answers:** Generate answers using retrieved government documents.
- **Interactive Chatbot:** Provide a simple user interface using Gradio.
- **Reduced Hallucinations:** Instruct the LLM to acknowledge when the available documents do not contain an answer.

## Tech Stack

- Python
- LangChain
- ChromaDB
- Sentence Transformers
- Ollama
- Large Language Models (LLMs)
- Gradio
- PDF document processing

## Project Structure

```text
government-services-rag/
├── documents/
│   └── government_schemes.pdf
├── chroma_db/
├── app.py
├── requirements.txt
└── README.md
```

## How It Works

1. **Document Collection:** Collect official government documents, scheme guidelines, and public-service information.
2. **Document Loading:** Load PDF and other supported documents using document loaders.
3. **Text Splitting:** Divide documents into smaller chunks for efficient retrieval.
4. **Embedding Generation:** Convert text chunks into vector embeddings using Sentence Transformers.
5. **Vector Storage:** Store embeddings and document metadata in ChromaDB.
6. **Query Processing:** Convert the user's question into an embedding.
7. **Semantic Retrieval:** Retrieve relevant document chunks from the vector database.
8. **Answer Generation:** Pass the retrieved context and user question to an LLM through Ollama.
9. **Response Display:** Display the answer through the Gradio interface, with source references where available.

## Example Questions

- What government schemes are available for students?
- What are the eligibility criteria for a particular scholarship?
- Which documents are required to apply for a government scheme?
- How can I apply for a government certificate?
- What is the application process for a pension scheme?
- Where can I find the official application website?

The answers depend on the government documents indexed in the system.

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd government-services-rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Ollama

Install Ollama from [ollama.com](https://ollama.com/download), then download a compatible language model.

```bash
ollama pull llama3.2:3b
```

Ensure the Ollama service is running before starting the application.

### 4. Add Government Documents

Place relevant government scheme guidelines, official PDFs, and public-service documents in the `documents/` directory.

Prefer official sources such as government portals and published scheme guidelines. Record the source URL and publication or update date when available.

### 5. Run the Application

```bash
python app.py
```

Open the local Gradio URL displayed in the terminal to interact with the chatbot.

## Reliability and Limitations

- Answers depend on the quality, completeness, and freshness of the indexed documents.
- Eligibility guidance should be treated as informational, not as an official determination of eligibility.
- Government rules and application procedures can change; verify important details against the current official source.
- The chatbot cannot submit applications, issue certificates, or access private government records unless those capabilities are separately implemented and authorized.
- Source citations should be displayed whenever possible so users can verify the information independently.

## Future Improvements

- Support multiple Indian languages, including Telugu, Hindi, and English.
- Add state-specific and central government service collections.
- Display source URLs, page numbers, and document update dates.
- Implement automated document updates from approved official sources.
- Add filters for state, department, scheme, and service category.
- Improve retrieval using hybrid search and reranking.
- Add a feedback mechanism to identify incorrect or outdated answers.
- Build a mobile-friendly interface.

## Contributing

Contributions are welcome. Fork the repository, create a feature branch, and submit a pull request with your changes.

## License

Choose and add an appropriate open-source license before distributing this project.

---

**Disclaimer:** This chatbot is an informational tool and is not an official government service. Users should verify eligibility, deadlines, fees, and application procedures through the relevant official government portal.