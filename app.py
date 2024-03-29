from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)

# Initialize components needed for PDF processing and QA
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embeddings = CohereEmbeddings(model="embed-english-light-v2.0", cohere_api_key="6k408g3yPCEhAZGpdq4RlO0Dd961ouv1ibC5ZVwH")
texts = []
model = "embed-english-light-v2.0"
rqa = None  # Initialize RetrievalQA instance as None initially

@app.route('/')
def home():
    return "Welcome to PDF-QA"

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    global rqa, texts

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    file_data = file.read()  # Read the file data
    # Process the PDF file
    raw_text = ""
    with io.BytesIO(file_data) as pdf_stream:
        doc_reader = PdfReader(pdf_stream)
        for page in doc_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

    # Split the text into chunks
    texts = text_splitter.split_text(raw_text)

    # Initialize the vector store with the texts
    docsearch = FAISS.from_texts(texts[:250], embeddings)
    print(docsearch.embedding_function)
    
    # Initialize the retriever outside the route functions
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})

    # Initialize the RetrievalQA instance
    global model, rqa
    rqa = RetrievalQA.from_chain_type(llm=embeddings,
                                          chain_type="stuff",
                                          retriever=retriever,
                                          return_source_documents=True)

    return jsonify({'message': 'PDF uploaded and processed successfully'})

@app.route('/question-answer', methods=['POST'])
def question_answer():
    global rqa
    data = request.json
    if 'question' not in data:
        return jsonify({'error': 'No question provided'})

    question = data['question']

    if rqa is None:
        return jsonify({'error': 'RetrievalQA not set up'})

    # Perform question answering using RetrievalQA
    answer = rqa(question)

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
