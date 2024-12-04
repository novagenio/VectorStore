import key_param

def load_pdf_file(path_pdf):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(key_param.path_pdf)
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
    function_embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key, \
    model=model)
    return(function_embeddings)

def vectorstore_chroma(documents, embedding):
    # chroma
    from langchain_chroma import Chroma
    vectorstore = Chroma.from_documents(documents, embedding, \
    persist_directory=key_param.chroma_db)
    return()

def vectorstore_faiss(documents, embedding):
    # faiss
    from langchain_community.vectorstores import FAISS
    vectorstore = FAISS.from_documents(documents, embedding)
    vectorstore.save_local(key_param.faiss_db)
    return()

def vectorstore_mongo_atlas(documents, embedding):
    # mongodb
    from langchain_community.vectorstores import MongoDBAtlasVectorSearch
    from pymongo import MongoClient
    client = MongoClient(key_param.MONGO_URI)
    collection = client[key_param.MONGO_dbName][key_param.MONGO_collectionName]
    vectorStore = MongoDBAtlasVectorSearch.from_documents(documents, \
    embedding, collection=collection)
    return()

def vectorstore_qdrant(documents, embedding):
    # qdrant
    from langchain_qdrant import QdrantVectorStore
    qdrant = QdrantVectorStore.from_documents(
        documents, embedding,
        url=key_param.QDRANT_URL,
        prefer_grpc=True, api_key=key_param.qdrant_api_key,
        collection_name=key_param.QDRANT_collection_name,)   


def vectorstorage_elasticsearch(documents, embedding):
    # elasticsearch
    from langchain_elasticsearch import ElasticsearchStore
    elastic_vector_search = ElasticsearchStore(
        es_cloud_id=key_param.es_cloud_id,
        index_name=key_param.elastic_index_name,
        embedding=embedding,
        es_user=key_param.user_elastics,
        es_password=key_param.pwd_elastic,
    )
    elastic_vector_search.add_documents(documents=documents)

def vectorstorage_pinecone(documents, embedding):
    # pinecone
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=key_param.pinecone_api_key)
    import time
    index_name = key_param.pinecone_index_name  # change if desired
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536 , # dimencion to  text-embedding-ada-002
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    from langchain_pinecone import PineconeVectorStore
    vector_store = PineconeVectorStore(index=index, embedding=embedding)
    vector_store.add_documents(documents=documents)




data = load_pdf_file(key_param.path_pdf)
all_splits = splits_text(data)
function_embeddings = define_embedding(key_param.model)
vectorstore_chroma(documents=all_splits, embedding=function_embeddings)
#vectorstore_faiss(documents=all_splits, embedding=function_embeddings)
#vectorstore_mongo_atlas(documents=all_splits, embedding=function_embeddings)
#vectorstore_qdrant(documents=all_splits, embedding=function_embeddings)
#vectorstorage_elasticsearch(documents=all_splits, embedding=function_embeddings)
#vectorstorage_pinecone(documents=all_splits, embedding=function_embeddings)

#for splits in all_splits:   print(splits )
