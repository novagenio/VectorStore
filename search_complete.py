

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
    from langchain_qdrant import FastEmbedSparse, RetrievalMode
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    vectorStore = QdrantVectorStore.from_documents(
        docs, embedding=embedding, sparse_embedding=sparse_embeddings,
        location=":memory:", collection_name="my_documents",
        retrieval_mode=RetrievalMode.HYBRID,)
    return(vectorStore)

def get_similaruty_search(vectorStore, query):
    docs = vectorStore.similarity_search(query)
    similarity_output = docs[0].page_content
    return(similarity_output) 

def llm_to_use():
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
        {"role": "system", "content": "resprespond using only this information:" + context},
        {"role": "user", "content": query}
      ]
    )
    print("content:", key_param.prompt + ' ' + context)
    return(completion.choices[0].message)

############ main
query = "Who can add or modify the above conditions through delegated acts"
embedding = define_embedding()
#vectorStore = get_vectorStorage_faiss(embedding)
#vectorStore = get_vectorStorage_chroma(embedding)
#vectorStore = get_vectorStorage_mongo(embedding)
vectorStore = get_vectorStorage_qdrant(embedding)
similarity_output = get_similaruty_search(vectorStore, query)

llm = llm_to_use()
retriever_output = RetrievalQA_function(embedding, vectorStore)

print('similarity_output o contexto: ', similarity_output)
print('')
print('retriever_output: ', retriever_output)

print('Generative')
print('primera respeusta generativa: ', get_generative_answer(similarity_output, query))

