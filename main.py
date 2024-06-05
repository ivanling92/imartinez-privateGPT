#!/usr/bin/env python3
import os
import argparse
from telnetlib import DEBUGLEVEL
import time
import glob
from typing import List
from multiprocessing import Pool
import streamlit as st
from tqdm import tqdm
import chromadb
import chromadb
from chromadb.api.segment import API

#Small agents handler
from groq import Groq
import ollama

#PDF Handlers
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract



from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All, LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document


DEBUG = True
OFFLINE = False

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# Initialize the Groq API
groq_api = st.secrets["GROQ"]

client = Groq(
    api_key=groq_api,
)

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

# Load environment variables
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY','source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50
from constants import CHROMA_SETTINGS



# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True)
        )
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.upper()}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)
    print(f"Split into {len(documents)} chunks of text (max. {chunk_size} tokens each)")
    return documents

def batch_chromadb_insertions(chroma_client: API, documents: List[Document]) -> List[Document]:
    """
    Split the total documents to be inserted into batches of documents that the local chroma client can process
    """
    # Get max batch size.
    max_batch_size = chroma_client.max_batch_size
    for i in range(0, len(documents), max_batch_size):
        yield documents[i:i + max_batch_size]


def does_vectorstore_exist(persist_directory: str, embeddings: HuggingFaceEmbeddings) -> bool:
    """
    Checks if vectorstore exists
    """
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    if not db.get()['documents']:
        return False
    return True


def ingest():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    # Chroma client
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)

    if does_vectorstore_exist(persist_directory, embeddings):
        # Update and store locally vectorstore
        st.write(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
        collection = db.get()
        documents = process_documents([metadata['source'] for metadata in collection['metadatas']])
        st.write(f"Creating embeddings. May take some minutes...")
        for batched_chromadb_insertion in batch_chromadb_insertions(chroma_client, documents):
            db.add_documents(batched_chromadb_insertion)
    else:
        # Create and store locally vectorstore
        st.write("Creating new vectorstore")
        documents = process_documents()
        st.write(f"Creating embeddings. May take some minutes...")
        # Create the db with the first batch of documents to insert
        batched_chromadb_insertions = batch_chromadb_insertions(chroma_client, documents)
        first_insertion = next(batched_chromadb_insertions)
        db = Chroma.from_documents(first_insertion, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, client=chroma_client)
        # Add the rest of batches of documents
        for batched_chromadb_insertion in batched_chromadb_insertions:
            db.add_documents(batched_chromadb_insertion)
    st.write(f"Ingestion complete! You can now use the chat interface to query your documents")
    return

def runQuery(prompt):
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    query = prompt
    res = qa(query)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents']
    result = answer + "The following document was referenced: "

    for document in docs:
        result += "\n" + document.metadata["source"] + ":" + document.page_content
         
    return result

def parse_arguments():
    parser = argparse.ArgumentParser(description='Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()



# PDF Checking Utilities
def is_likely_scanned_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        page_obj = pdf_reader.pages[0]  # Check the first page

        # Check for text content
        text = page_obj.extract_text()
        if not text.strip():  # Empty or just whitespace
            return True  # Likely scanned

        # Check for font information (often missing in scanned PDFs)
        if "/Font" not in page_obj.get_object():
            return True  # Likely scanned

        # Additional checks (optional):
        # - Look for embedded images
        # - Analyze the number of images vs. text content

        return False  # Not likely scanned

    except:
        return False  # Not a valid PDF at all



def pdf_to_image(pdf_file):
    # Convert all pages of the PDF into images
    pdf_binary = pdf_file.read()
    images = convert_from_bytes(pdf_binary)
    return images
    

def groq_process(text):
    userPrompt = {"role": "user", "content": text}
    systemPrompt = {"role": "system", "content": "You will be given chunks of text from an OCR output. You will try your best to correct typos and complete sentences in the text. Do not give any comments, just the corrected texts."}
    messageLog = [systemPrompt, userPrompt]

    try:
        chat_completion = client.chat.completions.create(
            messages= messageLog,
            model="Llama3-8b-8192",
        )

    except:
        return "Failed"
    return chat_completion.choices[0].message.content

def vainos_process(text):
    userPrompt = {"role": "user", "content": text}
    systemPrompt = {"role": "system", "content": "You will be given chunks of text from an OCR output. You will try your best to correct typos and complete sentences in the text.Do not give any comments, just the corrected texts."}

    try:
        response = ollama.chat(model='llama3', messages=[systemPrompt, userPrompt])
        return response['message']['content']

    except:
        return "Failed"

def extract_text_from_pdf(file):
    images = pdf_to_image(file)
    count = 0
    file_content = ""
    for image in images:
        count+=1
        if DEBUG:
            st.image(image)
        text = pytesseract.image_to_string(image)  # Extract text from the image using Tesseract
        if DEBUG:
            st.write(text)
        try:
            if OFFLINE:
                text = vainos_process(text)
            else:
                text = groq_process(text)
        except Exception as e:
            st.write("AI system offline, using raw text. Error: " + str(e))
        
        try:
            file_content += "\nPage: "+str(count)+"\n"+text+"\n"
        except:
            st.error("Error processing this page!")

    return file_content


def generate_response(prompt):
    return runQuery(prompt)
    #Can use groq to run another filter or guardrails if needed


def start_chat():
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you? You can ask me anything about the uploaded files."}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"], unsafe_allow_html=True)
    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt) 
                st.write(response, unsafe_allow_html=True) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


#Set a variable to store first run
if "chat" not in st.session_state.keys():
    st.session_state.chat = False
    
    
def main():
    st.title("Upload your PDFs to chat")
    uploaded_files = st.file_uploader("Choose a file", type=["txt", "pdf", "csv"], accept_multiple_files=True)
    
    if DEBUG:
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                # File details
                file_details = {
                    "filename": uploaded_file.name,
                    "filetype": uploaded_file.type,
                    "filesize": uploaded_file.size
                }
                st.write(file_details)
    
    # Processing the content. Check if it is scanned document or not
    file_processed = False            
    for file in uploaded_files:
        if file.type == "application/pdf":
            #check if it is scanned
            scanned = is_likely_scanned_pdf(file)
            if scanned:
                if DEBUG:
                    st.write("Scanned PDF")
                #Run OCR and extract text
                    file_content = extract_text_from_pdf(file)
                    if DEBUG:
                        st.write(file_content)
                    #Write file content to a txt-file in source_documents folder
                    with open("source_documents/"+file.name+".txt", "w") as text_file:
                        text_file.write(file_content)
                    
            else:
                if DEBUG:
                    st.write("Not Scanned PDF")
                with open("source_documents/"+file.name, "wb") as f:
                    f.write(file.getbuffer())
                    if DEBUG:
                        st.write("File received")
                
        else:
            #copy the file and paste it to source_documents
            with open("source_documents/"+file.name, "wb") as f:
                f.write(file.getbuffer())
                if DEBUG:
                    st.write("File received")
    file_processed = True
    if file_processed and st.button("Click to start reading the documents"):
        st.write("AI is reading the documents now. Please be patient!")
        with st.spinner('Reading'):
            ingest()
        st.success("AI has read the documents successfully!")
        
    start_chat()

            

if __name__ == "__main__":
    main()