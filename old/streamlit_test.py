from langchain_community.chat_models import ChatOllama
from langchain.embeddings.ollama import OllamaEmbeddings
import streamlit as st

# import PDF loaders
from langchain.document_loaders.pdf import PyPDFLoader
# import chroma as vector store
from langchain.vectorstores.chroma import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


@st.cache_resource
def get_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        cache_dir='./model/',
        
    )

llm = ChatOllama(model="analyzer_llama3", temperature=0.9) 
embeddings = OllamaEmbeddings()

loader = PyPDFLoader('data/sec1.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, embeddings, collection_name='sec1')

# create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="sec",
    description="sec filing info",
    vectorstore=store
)
# convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
# add the toolkit to an end to end langchain
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbox=True
)

#llm = ollama(temperature=0.9)
prompt = st.text_input('input prompt here')
# llm_chain = prompt | llm
if prompt:
    # response = llm_chain.invoke()
    #response = llm.invoke(prompt) 
    response = agent_executor.run(prompt)
    st.write(response.content) 

    with st.expander('Document Similarity Search'):
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)