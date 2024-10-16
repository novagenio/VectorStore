import key_param

def load_pdf_file(path_pdf):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader('pdf/eu_ai_act.pdf')
    documents = loader.load()
    return documents

def splits_text(data):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(data)
    return(all_splits)

def define_embedding(model):
    from langchain_community.llms import OpenAI
    from langchain_openai import OpenAIEmbeddings
    function_embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key, model=model)
    return(function_embeddings)

def vectorstore_chroma(documents, embedding):
    from langchain_chroma import Chroma
    vectorstore = Chroma.from_documents(documents, embedding, persist_directory=key_param.chroma_db)
    return()

def vectorstore_faiss(documents, embedding):
    from langchain_community.vectorstores import FAISS
    vectorstore = FAISS.from_documents(documents, embedding)
    vectorstore.save_local(key_param.faiss_db)
    return()

def vectorstore_mongo_atlas(documents, embedding):
    from langchain_community.vectorstores import MongoDBAtlasVectorSearch
    from pymongo import MongoClient
    client = MongoClient(key_param.MONGO_URI)
    collection = client[key_param.MONGO_dbName][key_param.MONGO_collectionName]
    vectorStore = MongoDBAtlasVectorSearch.from_documents(documents, embedding, collection=collection)
    return()

def vectorstore_qdrant(documents, embedding):
    from langchain_qdrant import QdrantVectorStore
    qdrant = QdrantVectorStore.from_documents(
        documents, embedding,
        url=key_param.QDRANT_URL,
        prefer_grpc=True, api_key=key_param.qdrant_api_key,
        collection_name=key_param.QDRANT_collection_name,)   



data = load_pdf_file(key_param.path_pdf)
all_splits = splits_text(data)
function_embeddings = define_embedding(key_param.model)
vectorstore_chroma(documents=all_splits, embedding=function_embeddings)
vectorstore_faiss(documents=all_splits, embedding=function_embeddings)
vectorstore_mongo_atlas(documents=all_splits, embedding=function_embeddings)
vectorstore_qdrant(documents=all_splits, embedding=function_embeddings)
