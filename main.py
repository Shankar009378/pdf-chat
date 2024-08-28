from fastapi import FastAPI, File, UploadFile, HTTPException
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()


origins = [
    '*'  # Add your frontend's URL here
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

key = os.getenv("API_KEY")
genai.configure(api_key=key)


def get_pdf_text(pdf_docs: List[UploadFile]):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks: List[str]):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


@app.post("/upload-pdfs/")
async def upload_pdfs(pdf_docs: List[UploadFile] = File(...)):
    if not pdf_docs:
        raise HTTPException(status_code=400, detail="No PDF files uploaded")
    
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    
    return {"message": "PDFs processed and vector store created successfully"}


class QuestionRequest(BaseModel):
    user_question: str

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    user_question = request.user_question
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return {"answer": response["output_text"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
