import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain




from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


# Step 1: Setup Groq LLM
GROQ_API_KEY= os.environ.get('GROQ_API_KEY')
GROQ_MODEL_NAME= "llama-3.1-8b-instant" # change to any supported Groq model
# Debugging check:
if GROQ_API_KEY is None:
    print("ERROR: GROQ_API_KEY not found! Check your .env file.")
else:
    print("Success: API Key loaded.")
llm= ChatGroq(
    model=GROQ_MODEL_NAME,
    temperature=0.5,
    max_tokens=512,
    groq_api_key=GROQ_API_KEY,
)


# Step 2: Connect LLM with FAISS and Create chain
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# step 3 : build RAG chain
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# document combiner chain (stuff documents into prompt)
combine_docs_chain = create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)

#Retrieval chain (retriever + doc combiner)
#rag_chain = create_stuff_documents_chain(db.as_retriever(search_kwargs={'k':3}),combine_docs_chain)

rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k':3}), combine_docs_chain)
# Now invoke with a single query
user_query=input("Write Query Here: ")
response=rag_chain.invoke({'input': user_query})
print("RESULT: ", response["answer"])
print("\nSOURCE DOCUMENTS:")
for doc in response ["context"]:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")
