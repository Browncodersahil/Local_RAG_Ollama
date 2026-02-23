from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma 
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("Restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
 
db_location = "./chroma_db"
add_docs = not os.path.exists(db_location)

if add_docs:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
           page_content=row["Title"] + " " + row["Review"],
           metadata={"rating": row["Rating"], "date": row["Date"]},
           id=str(i)
        ) 
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_docs:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)