from PyPDF2 import PdfReader

#uploading pdf
doc_reader = PdfReader("./impromptu-rh.pdf")

#parsing
raw_text = ""
for i,page in enumerate(doc_reader.pages):
  text = page.extract_text()
  if text:
    raw_text += text
    
#chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                               chunk_overlap = 200,
                                               length_function = len)
texts = text_splitter.split_text(raw_text)

#create embedding
from langchain.embeddings import CohereEmbeddings
embeddings = CohereEmbeddings(
                model="embed-english-light-v2.0",
                cohere_api_key="6k408g3yPCEhAZGpdq4RlO0Dd961ouv1ibC5ZVwH"
            )

vectors = embeddings.embed_documents(
    [
        "hi there",
        "i love machine learning",
        "india is my country",
        "engineering college palakkad"
    ]
)
print(len(vectors))
print(len(vectors[0]))

embedd_query = embeddings.embed_query("What are the types of machine learning")
embedd_query

#Store data in VectorDB
from langchain.vectorstores import FAISS
docsearch = FAISS.from_texts(texts[:250],embeddings)
docsearch.embedding_function

#Semantic Search
query = "how does GPT-4 change social media"
docs = docsearch.similarity_search(query)

from langchain.llms import HuggingFaceHub
model = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", huggingfacehub_api_token="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")

#load_qa
# from langchain.chains.question_answering import load_qa_chain
# chain = load_qa_chain(model,
#                       chain_type="refine"
#                       )

# query = "how much data producing daily"
# docs = docsearch.similarity_search(query)
# print(chain.run(input_documents = docs, question = query))

#Retrieval QA
from langchain.chains import RetrievalQA
# set up FAISS as a generic retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})
# create the chain to answer questions
rqa = RetrievalQA.from_chain_type(llm=model,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

query = "what is OpenAI?"
print(rqa(query)["result"])