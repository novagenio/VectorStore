import warnings
warnings.filterwarnings("ignore")

import key_param

def define_embedding():
    from langchain_openai import OpenAIEmbeddings
    import key_param
    embedding = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
    return(embedding)

def get_vectorStorage_faiss(embedding):
    from langchain_community.vectorstores import FAISS
    vectorStore = FAISS.load_local(key_param.faiss_db, embedding, allow_dangerous_deserialization=True)
    return(vectorStore)

def get_vectorStorage_chroma(embedding):
    from langchain_chroma import Chroma
    vectorStore = Chroma(persist_directory=key_param.chroma_db, embedding_function=embedding)
    return(vectorStore) 

def get_vectorStorage_mongo(embedding):
    from pymongo import MongoClient
    from langchain_mongodb import MongoDBAtlasVectorSearch
    client = MongoClient(key_param.MONGO_URI)
    collection = client[key_param.MONGO_dbName][key_param.MONGO_collectionName]
    vectorStore = MongoDBAtlasVectorSearch(collection, embedding)
    return(vectorStore) 

def get_vectorStorage_qdrant(embedding):
    from langchain_qdrant import QdrantVectorStore
    qdrant = QdrantVectorStore.from_existing_collection(embedding=embedding, collection_name=key_param.QDRANT_collection_name, url=key_param.QDRANT_URL,)
    return(vectorStore)


def get_vectorstorage_elasticsearch(embedding):
    from langchain_elasticsearch import ElasticsearchStore
    vectorStore = ElasticsearchStore(
        es_cloud_id=key_param.es_cloud_id,
        index_name=key_param.elastic_index_name,
        embedding=embedding,
        es_user=key_param.user_elastics,
        es_password=key_param.pwd_elastic,
    )
    return(vectorStore)

def get_vectorstorage_elasticsearch_docker(embedding):
    from langchain_elasticsearch import ElasticsearchStore
    vectorStore = ElasticsearchStore(
        es_url=key_param.elastic_url_docker,
        index_name=key_param.elastic_index_name,
        embedding=embedding,
    )
    return(vectorStore)

def get_vectorstorage_pinecone(embedding):
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=key_param.pinecone_api_key)
    import time
    index_name = key_param.pinecone_index_name  # change if desired
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name, dimension=1536 , # dimencion to  text-embedding-ada-002
            metric="cosine",  spec=ServerlessSpec(cloud="aws", region="us-east-1"),)
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    index = pc.Index(index_name)
    from langchain_pinecone import PineconeVectorStore
    vector_store = PineconeVectorStore(index=index, embedding=embedding)
    return(vector_store)
    

def get_similarity_search(vectorStore, query):
    docs = vectorStore.similarity_search(query)
    similarity_output = docs[0].page_content
    return(similarity_output) 

def define_language_generation_model():
    from langchain_openai import OpenAI
    import key_param
    llm = OpenAI(openai_api_key=key_param.openai_api_key, temperature=0)
    return(llm)

def RetrievalQA_function(embedding, vectorStore):
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 6}, embedding_function=embedding)
    from langchain.chains import RetrievalQA
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.invoke(query)
    return(retriever_output)

def get_generative_answer(similarity_output, query):
    from openai import OpenAI
    client = OpenAI(api_key=key_param.openai_api_key)
    context=similarity_output
    completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "system", "content": "Respond using only the follow information:" + context},
        {"role": "user", "content": query}
      ]
    )
    print("content:", key_param.prompt + ' ' + context)
    return(completion.choices[0].message)

############ main

query = "How it will be enforced and what are the penalties?"

embedding = define_embedding()
#vectorStore = get_vectorStorage_faiss(embedding)
#vectorStore = get_vectorstorage_pinecone(embedding)
#vectorStore = get_vectorStorage_mongo(embedding)
#vectorStore = get_vectorStorage_qdrant(embedding)
#vectorStore = get_vectorstorage_elasticsearch(embedding)
vectorStore = get_vectorstorage_elasticsearch_docker(embedding)

#vectorStore = get_vectorstorage_pinecone(embedding)


similarity_output = get_similarity_search(vectorStore, query)
llm = define_language_generation_model()

print('similarity_output: ', similarity_output)
print('')
print('Generative')
print('first generative answer ', get_generative_answer(similarity_output, query))

