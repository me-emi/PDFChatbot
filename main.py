# python version==3.10.7
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import validators
from validators import ValidationFailure
import streamlit as st


def is_string_an_url(url_string: str) -> bool:
    url_result = validators.url(url_string)
    if isinstance(url_result, ValidationFailure):
        return False
    return url_result


url = st.text_input('Enter the link to the PDF', '')  # https://www.oecd.org/education/1841883.pdf
if not url:
    st.warning("Please put a valid link!")
else:
    if is_string_an_url(url) and "pdf" in url:
        question = st.text_input('Enter the question that you have', '')
        if not question:
            st.warning("Please fill out the question!")
        else:
            loader = PyPDFLoader(url)  # PyPDFLoader("https://www.oecd.org/education/1841883.pdf")
            documents = loader.load_and_split()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
            texts = text_splitter.split_documents(documents)
            text_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma.from_documents(texts, text_embeddings, persist_directory="db")
            model_n_ctx = 1000
            model_path = "./ggml-gpt4all-j-v1.3-groovy.bin"
            llm = GPT4All(model=model_path, n_ctx=1000, backend="gptj", verbose=False)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 2}),
                return_source_documents=True,
                verbose=False,
            )

            result = qa(
                question  # it has to be a string
            )
            st.write('The answer is:', result["result"])
            st.write(
                "[Emilija Gjorgjevska, Business Software Engineer](https://www.linkedin.com/in/emilijagjorgjevska/)")
    else:
        st.warning("Your link is invalid, please re-enter a valid PDF link!")
