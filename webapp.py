import streamlit as st
from constants import LLM_LLAMA_2
from pdf_qa import PdfQA
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import shutil
from constants import *
import os


# Streamlit app code
st.set_page_config(
    page_title='maya.ai PDF Q&A Bot',
    page_icon='🔖',
    layout='wide',
    initial_sidebar_state='auto',
)


if "pdf_qa_model" not in st.session_state:
    st.session_state["pdf_qa_model"]:PdfQA = PdfQA() ## Intialisation

## To cache resource across multiple session 
#@st.cache_resource
def load_llm(llm,load_in_8bit):

    if llm == LLM_OPENAI_GPT35:
        return PdfQA.create_openai()
    elif llm == LLM_LLAMA_2:
        return PdfQA.create_llama_13b()
    elif llm == LLM_FLAN_T5_SMALL:
        return PdfQA.create_flan_t5_small(load_in_8bit)
    elif llm == LLM_FLAN_T5_BASE:
        return PdfQA.create_flan_t5_base(load_in_8bit)
    elif llm == LLM_FLAN_T5_LARGE:
        return PdfQA.create_flan_t5_large(load_in_8bit)
    elif llm == LLM_FLAN_T5_XL:
        return PdfQA.create_flan_t5_xl(load_in_8bit)
    elif llm == LLM_FALCON_SMALL:
        return PdfQA.create_falcon_instruct_small(load_in_8bit)
    elif llm == LLM_FALCON_7B_LIGHTWEIGHT:
        return PdfQA.create_falcon_7b_lightweight(load_in_8bit)
    elif llm == LLM_FALCON_7B:
        return PdfQA.create_falcon_7b(load_in_8bit)
    else:
        raise ValueError("Invalid LLM setting")

## To cache resource across multiple session
#@st.cache_resource
def load_emb(emb):
    if emb == EMB_INSTRUCTOR_XL:
        return PdfQA.create_instructor_xl()
    elif emb == EMB_SBERT_MPNET_BASE:
        return PdfQA.create_sbert_mpnet()
    elif emb == EMB_SBERT_MINILM:
        pass ##ChromaDB takes care
    elif emb == EMB_OPENAI_ADA:
        pass ##OpenAI takes care
    else:
        raise ValueError("Invalid embedding setting")



st.title("maya.ai PDF Q&A Bot")

with st.sidebar:
    #emb = st.radio("**Select Embedding Model**", [EMB_INSTRUCTOR_XL, EMB_SBERT_MPNET_BASE,EMB_SBERT_MINILM,EMB_OPENAI_ADA],index=1)
    llm = st.radio("**Select LLM Model**", [LLM_OPENAI_GPT35, LLM_FLAN_T5_SMALL,LLM_FLAN_T5_BASE,LLM_FLAN_T5_LARGE,LLM_FLAN_T5_XL,LLM_FALCON_7B_LIGHTWEIGHT, LLM_FALCON_7B],index=2)
    pdf = st.radio("**Select PDF Document**", ["ADIB_COVEREDCARD_POLICY","UAE_CUSTOMER_PROTECTION_REGULATION",
                                               "CREDIT_POLICY_DOCUMENT","BANK_COMPLIANCE_POLICY_INDIA",
                                               "SIB_LOAN_AGREEMENT","**LOAD_FROM_LOCAL**"])
    pdf_file = st.file_uploader("**Upload PDF**", type="pdf")
    load_in_8bit=False
    target_directory = "/tmp/"
    target_path = None
    invalid = False
    if st.button("Submit") and pdf is not None:
        with st.spinner(text="Uploading PDF and Generating Embeddings.."):
            if pdf == "**LOAD_FROM_LOCAL**":
                if pdf_file is None:
                    st.error("No Local File Selected")
                    invalid = True
                else:
                    print("file writing started")
                    filename = pdf_file.name
                    target_path = os.path.join(target_directory, filename)
                    with open(target_path, "wb") as f:
                        f.write(pdf_file.read())
                    print("file written")
            if not invalid:
                #emb = EMB_OPENAI_ADA if llm == LLM_OPENAI_GPT35 else EMB_SBERT_MPNET_BASE
                emb = EMB_SBERT_MINILM
                st.session_state["pdf_qa_model"].config = {
                    "pdf_path": target_path,
                    "embedding": emb,
                    "llm": llm,
                    "pdf": pdf,
                    "load_in_8bit": load_in_8bit
                }
                st.session_state["pdf_qa_model"].embedding = load_emb(emb)
                st.session_state["pdf_qa_model"].llm = load_llm(llm,load_in_8bit)        
                st.session_state["pdf_qa_model"].init_models()
                st.session_state["pdf_qa_model"].init_embeddings()
                st.session_state["pdf_qa_model"].vector_db_pdf()
                st.sidebar.success("PDF uploaded successfully")
                    

question = st.text_input('Ask a question', 'What is this document?')

if st.button("Answer"):
    try:
        st.session_state["pdf_qa_model"].retreival_qa_chain()
        answer = st.session_state["pdf_qa_model"].answer_query(question)
        st.write(f"{answer}")
    except Exception as e:
        st.error(f"Error answering the question: {str(e)}")