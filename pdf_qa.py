from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from constants import *
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import os
import re
import tiktoken
import glob
import os

import langchain

# loaders
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

# splits
from langchain.text_splitter import RecursiveCharacterTextSplitter

# prompts
from langchain import PromptTemplate, LLMChain

# vector stores
from langchain.vectorstores import FAISS

# models
from langchain.llms import HuggingFacePipeline
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import torch
import transformers

# retrievers
from langchain.chains import RetrievalQA

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import os
import glob
import textwrap
import time

os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY

class PdfQA:
    def __init__(self,config:dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None

    # The following class methods are useful to create global GPU model instances
    # This way we don't need to reload models in an interactive app,
    # and the same model instance can be used across multiple user sessions
    @classmethod
    def create_instructor_xl(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceInstructEmbeddings(model_name=EMB_INSTRUCTOR_XL, model_kwargs={"device": device})
    
    @classmethod
    def create_sbert_mpnet(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})    
    
    @classmethod
    def create_flan_t5_xxl(cls, load_in_8bit=False):
        print("Loading flan_t5_xxl model.....")
        # Local flan-t5-xxl with 8-bit quantization for inference
        # Wrap it in HF pipeline for use with LangChain
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-xxl",
            max_new_tokens=200,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    @classmethod
    def create_flan_t5_xl(cls, load_in_8bit=False):
        print("Loading flan_t5_xl model.....")
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-xl",
            max_new_tokens=200,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    
    @classmethod
    def create_flan_t5_small(cls, load_in_8bit=False):
        print("Loading flan_t5_small model.....")
        # Local flan-t5-small for inference
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    @classmethod
    def create_flan_t5_base(cls, load_in_8bit=False):
        print("Loading flan_t5_base model.....")
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    @classmethod
    def create_flan_t5_large(cls, load_in_8bit=False):
        print("... Loading flan_t5_large model.....")
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    # @classmethod
    # def create_fastchat_t5_xl(cls, load_in_8bit=False):
    #     print("Loading fastchat_t5_xl model.....")
    #     return pipeline(
    #         task="text2text-generation",
    #         model = "lmsys/fastchat-t5-3b-v1.0",
    #         max_new_tokens=100,
    #         model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
    #     )
        
    @classmethod
    def create_llama_13b(cls, load_in_8bit=False):
        print("Loading Llama-2-13b model.....")
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            model_path="./llama-2-13b-chat.ggmlv3.q4_0.bin",
            callback_manager=callback_manager,
            verbose=True,
        )
        
    @classmethod
    def create_openai(cls):
        print("Loading OpenAI GPT3.5 model.....")
        openai_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        return openai_llm

    # @classmethod
    # def create_falcon_7b_lightweight(cls,load_in_8bit=False):
    #     print("Loading falcon 7b lightweight model.....")
    #     #model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"
    #     model_id = "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2"
    #     tokenizer=AutoTokenizer.from_pretrained(model_id)
    #     model=AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    #     return pipeline(
    #     task = "text-generation",
    #     model = model,
    #     tokenizer = tokenizer,
    #     pad_token_id = tokenizer.eos_token_id,
    #     max_length = 1048,
    #     temperature = 0.01,
    #     top_p = 0.95,
    #     repetition_penalty = 1.15,
    #     trust_remote_code=True,
    #     do_sample=True,
    #     #device="cuda",
    #     device_map="auto"
    # )
        
    @classmethod
    def create_falcon_7b_lightweight(cls,load_in_8bit=False):
        print("Loading falcon 7b lightweight model.....")
        #model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"
        model_repo = 'vilsonrodrigues/falcon-7b-instruct-sharded'
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True)
        
        max_len = 1024
        
        return pipeline(
            task = "text-generation",
            model = model,
            tokenizer = tokenizer,
            pad_token_id = tokenizer.eos_token_id,
            max_length = max_len,
            temperature = 0,
            top_p = 0.95,
            repetition_penalty = 1.15
        )

    @classmethod
    def create_falcon_7b(cls,load_in_8bit=False):
        print("Loading falcon 7b.....")
        #model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"
        model_repo = 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2'
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            load_in_4bit=True,
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        max_len = 1024
        
        return pipeline(
            task = "text-generation",
            model = model,
            tokenizer = tokenizer,
            pad_token_id = tokenizer.eos_token_id,
            max_length = max_len,
            temperature = 0,
            top_p = 0.95,
            repetition_penalty = 1.15
        )
        
    @classmethod
    def create_falcon_7b_v2(cls,load_in_8bit=False):
        print("Loading falcon 7b tiiuae/falcon-7b.....")
        #model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"
        model_repo = 'tiiuae/falcon-7b'
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            load_in_4bit=True,
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        max_len = 1024
        
        return pipeline(
            task = "text-generation",
            model = model,
            tokenizer = tokenizer,
            pad_token_id = tokenizer.eos_token_id,
            max_length = max_len,
            temperature = 0,
            top_p = 0.95,
            repetition_penalty = 1.15
        )
    # @classmethod
    # def create_falcon_instruct_small(cls, load_in_8bit=False):
    #     print("Loading Falcon 7B model.....")
    #     model = "tiiuae/falcon-7b-instruct"

    #     tokenizer = AutoTokenizer.from_pretrained(model)
    #     hf_pipeline = pipeline(
    #             task="text-generation",
    #             model = model,
    #             tokenizer = tokenizer,
    #             trust_remote_code = True,
    #             max_new_tokens=100,
    #             model_kwargs={
    #                 "device_map": "auto", 
    #                 "load_in_8bit": load_in_8bit, 
    #                 "max_length": 512, 
    #                 "temperature": 0.01,
    #                 "torch_dtype":torch.bfloat16,
    #                 }
    #         )
    #     return hf_pipeline
    
    def init_embeddings(self) -> None:
        # OpenAI ada embeddings API
        if self.config["embedding"] == EMB_OPENAI_ADA:
            print("Loading OpenAI Embeddings...")
            self.embedding = OpenAIEmbeddings()
            print(self.embedding)
            print("OpenAi Ebeddings loaded")
        elif self.config["embedding"] == EMB_INSTRUCTOR_XL:
            # Local INSTRUCTOR-XL embeddings
            print("Loading hkunlp/instructor-xl Embeddings...")
            self.embedding = PdfQA.create_instructor_xl()
        elif self.config["embedding"] == EMB_SBERT_MPNET_BASE:
            ## this is for SBERT
            print("Loading sentence-transformers/all-mpnet-base-v2 Embeddings...")
            self.embedding = PdfQA.create_sbert_mpnet()
        else:
            self.embedding = None ## DuckDb uses sbert embeddings
            # raise ValueError("Invalid config")

    def init_models(self) -> None:
        """ Initialize LLM models based on config """
        load_in_8bit = self.config.get("load_in_8bit",False)
        # OpenAI GPT 3.5 API
        if self.config["llm"] == LLM_OPENAI_GPT35:
            self.llm = PdfQA.create_openai()
        elif self.config["llm"] == LLM_LLAMA_2:
            self.llm = PdfQA.create_llama_13b()
        elif self.config["llm"] == LLM_FLAN_T5_SMALL:
            if self.llm is None:
                self.llm = PdfQA.create_flan_t5_small(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FLAN_T5_BASE:
            if self.llm is None:
                self.llm = PdfQA.create_flan_t5_base(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FLAN_T5_LARGE:
            if self.llm is None:
                self.llm = PdfQA.create_flan_t5_large(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FLAN_T5_XL:
            if self.llm is None:
                self.llm = PdfQA.create_flan_t5_xl(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FLAN_T5_XXL:
            if self.llm is None:
                self.llm = PdfQA.create_flan_t5_xxl(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FASTCHAT_T5_XL:
            if self.llm is None:
                self.llm = PdfQA.create_fastchat_t5_xl(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FALCON_SMALL:
            if self.llm is None:
                self.llm = PdfQA.create_falcon_instruct_small(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FALCON_7B_LIGHTWEIGHT:
            if self.llm is None:
                self.llm = PdfQA.create_falcon_7b_lightweight(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FALCON_7B:
            if self.llm is None:
                self.llm = PdfQA.create_falcon_7b(load_in_8bit=load_in_8bit)
        else:
            raise ValueError("Invalid config")        
    def vector_db_pdf(self) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        pdf_path = self.config.get("pdf_path",None)
        persist_directory = self.config.get("persist_directory",None)
        pdf = self.config.get("pdf",None)
        if pdf_path and os.path.exists(pdf_path):
            tmp_directory = '/tmp'
            
            loader = DirectoryLoader(
                tmp_directory,
                glob="./*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True
            )

            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 800,
                chunk_overlap = 0
            )

            texts = text_splitter.split_documents(documents)


            embeddings = HuggingFaceInstructEmbeddings(
                model_name = self.config["embedding"],
                model_kwargs = {"device": "cuda"}
            )
            
            # embeddings = OpenAIEmbeddings()

            ### create embeddings and DB
            self.vectordb = FAISS.from_documents(
                documents = texts, 
                embedding = embeddings
            )
            
            ### persist vector database
            # self.vectordb.save_local("SIB_LOAN_AGREEMENT")
            
            #Delete PDF FIles after loading to vector DB
            pdf_files = glob.glob(os.path.join(tmp_directory, '*.pdf'))
            for pdf_file in pdf_files:
                os.remove(pdf_file)
        elif pdf:
            embeddings = HuggingFaceInstructEmbeddings(
                model_name = self.config["embedding"],
                model_kwargs = {"device": "cuda"}
            )
            
            # embeddings = OpenAIEmbeddings()
            
            if pdf == "SIB_LOAN_AGREEMENT":
                self.vectordb = FAISS.load_local("SIB_LOAN_AGREEMENT/",embeddings)
            elif pdf == "ADIB_COVEREDCARD_POLICY":
                self.vectordb = FAISS.load_local("ADIB_COVEREDCARD_POLICY/",embeddings)
            elif pdf == "BANK_COMPLIANCE_POLICY_INDIA":
                self.vectordb = FAISS.load_local("BANK_COMPLIANCE_POLICY_INDIA/",embeddings)
            elif pdf == "CREDIT_POLICY_DOCUMENT":
                self.vectordb = FAISS.load_local("CREDIT_POLICY_DOCUMENT/",embeddings)
            elif pdf == "UAE_CUSTOMER_PROTECTION_REGULATION":
                self.vectordb = FAISS.load_local("UAE_CUSTOMER_PROTECTION_REGULATION/",embeddings)
        else:
            raise ValueError("NO PDF found")

    def retreival_qa_chain(self):
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        question_template = """
                You are chatbot to read the given context and answer for questions.
                Don't try to make up an answer, if you don't know just say that you don't know.
                Answer in the same language the question was asked.
                Use only the following pieces of context to answer the question at the end.

                {context}

                Question: {question}
                Answer:"""
        prompt_template = PromptTemplate(
            template=question_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": prompt_template}
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k":3, "search_type" : "similarity"})
        
        if self.config["llm"] == LLM_OPENAI_GPT35:
          # Use ChatGPT API
          self.qa = RetrievalQA.from_chain_type(
                        llm=OpenAI(model_name=LLM_OPENAI_GPT35, temperature=0.), 
                        chain_type="stuff",
                        retriever=self.retriever , 
                        chain_type_kwargs=chain_type_kwargs
                        )
          self.qa.combine_documents_chain.verbose = True
          self.qa.return_source_documents = True
        else:
            hf_llm = HuggingFacePipeline(pipeline = self.llm)
            #hf_llm = HuggingFacePipeline(pipeline=self.llm,model_id=self.config["llm"])

            self.qa = RetrievalQA.from_chain_type(
                llm = hf_llm,
                chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
                retriever = self.retriever, 
                chain_type_kwargs = chain_type_kwargs,
                return_source_documents = True,
                verbose = False
            )
    
    def wrap_text_preserve_newlines(self, text, width=700):
        # Split the input text into lines based on newline characters
        lines = text.split('\n')

        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)

        return wrapped_text


    def process_llm_response(self, llm_response):
        ans = self.wrap_text_preserve_newlines(llm_response['result'])
        
        sources_used = ' \n\n'.join(
            [
                source.metadata['source'].split('/')[-1][:-4] + ' - page: ' + str(source.metadata['page'])
                for source in llm_response['source_documents']
            ]
        )
        
        ans = ans + '\n\nSources: \n\n' + sources_used
        return ans
    
    # def llm_ans(query):
    #     start = time.time()
    #     llm_response = self.qa(query)
    #     ans = process_llm_response(llm_response)
    #     end = time.time()

    #     time_elapsed = int(round(end - start, 0))
    #     time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    #     return ans + time_elapsed_str
    
    def answer_query(self,question:str) ->str:
            """
            Answer the question
            """
            start = time.time()
            llm_response = self.qa(question)
            ans = self.process_llm_response(llm_response)
            end = time.time()

            time_elapsed = int(round(end - start, 0))
            time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
            return ans + time_elapsed_str