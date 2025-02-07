import os
import time
from operator import itemgetter
from typing import List, Tuple

import google.auth
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from google.generativeai.types import HarmBlockThreshold, HarmCategory

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import LLMResult
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable

# Contextual Compression & Reranking
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# LangChain LLMs & Vector Store
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# Google AI & Vertex AI
from langchain_google_community import VertexAISearchRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

# RAGAS Components
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

##############

def chunk_docs_recursive(docs: list, chunk_size=500, chunk_overlap=50) -> list:
    """
    Chunk the docs using a recursive chunker
    https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
    """

    recursive_chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = recursive_chunker.split_documents(docs)

    return chunks

def chunk_docs_semantic(docs: list, embeddings: Embeddings, breakpoint_threshold_type: str) -> list:
    """
    Chunk the docs using a semantic chunker
    breakpoint_threshold_type = percentile, standard_deviation, interquartile, or gradient
    """

    semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type=breakpoint_threshold_type)

    chunks = semantic_chunker.split_documents(docs)

    return chunks

def create_chain_openai(model: str, prompt_template: ChatPromptTemplate, retriever: BaseRetriever, use_cohere: bool = False) -> RunnableSerializable:
    """
    Create a Langchain chain using OpenAI
    https://python.langchain.com/docs/how_to/contextual_compression/
    """     

    llm = ChatOpenAI(model=model)

    if use_cohere:
        compressor = CohereRerank(model="rerank-english-v3.0")
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

        chain = (
            {"context": itemgetter("question") | compression_retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": prompt_template | llm, "context": itemgetter("context")}
            )
    else:        
        chain = (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")} 
            | RunnablePassthrough.assign(context=itemgetter("context")) 
            | {"response": prompt_template | llm, "context": itemgetter("context")}
            )

    return chain

def create_chain_vertexai (model_name: str, prompt_template: ChatPromptTemplate, retriever: BaseRetriever) -> RunnableSerializable:

    """
    # Create a Langchain chain using Google Gemini 
    # https://python.langchain.com/docs/integrations/llms/google_ai/
    # https://python.langchain.com/docs/integrations/chat/google_generative_ai/
    # https://ai.google.dev/gemini-api/docs/safety-settings 
    """

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        )
  
    chain = (
        {"context": itemgetter("question") | retriever, 
        "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context")) | 
        {"response": prompt_template | llm,
        "context": itemgetter("context")}
        )

    return chain

def create_chat_prompt_template(template: str = None) -> ChatPromptTemplate:
    """
    # Create a prompt template
    # https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/#chatprompttemplate
    # https://python.langchain.com/v0.2/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
    """
    
    if template is None:
        template = """
        You are an expert assistant designed to help users analyze and answer questions about 10K annual reports filed by publicly traded companies. Users may ask about specific sections, financial metrics, trends, or comparisons across multiple reports. Your role is to provide accurate, concise, and relevant answers, referencing appropriate sections or data points where applicable.

        When responding, adhere to the following principles:

        Understand the Question: Identify whether the query is focused on a specific company, year, or metric, or if it spans multiple reports for comparison.
        Clarify Uncertainty: If a user's question is unclear, ask for clarification or additional context.
        Locate and Reference Information: Use relevant sections of the 10K report(s), such as MD&A, Financial Statements, Risk Factors, or Notes to Financial Statements, to back up your answers.
        Synthesize Data: Provide summaries or insights when the question involves comparing data or trends across multiple reports.
        Stay Objective: Avoid providing subjective opinions or interpretations beyond the factual content in the reports.
        Example User Queries and Expected Responses:

        "What was the revenue for Company X in 2022 and 2023?"

        Locate and report the revenue figures from the Income Statements of the respective 10K reports for 2022 and 2023.
        "What are the main risk factors for Company Y in its latest report?"

        Summarize the key risk factors from the most recent 10K report's "Risk Factors" section.
        "How did the operating income of Company Z change over the last three years?"

        Extract operating income figures from the 10K reports for the past three years and provide a brief comparison.
        "Compare the debt levels of Company A and Company B in 2023."

        Retrieve debt-related figures from the Balance Sheets or Notes to Financial Statements of both companies and summarize the comparison.
        "What trends are evident in Company W's R&D expenses over the last five years?"

        Summarize trends using data from the Income Statements or footnotes for R&D expenses across five consecutive 10K reports.
        Assumptions and Constraints:

        Only use inforomation from 10K reports provided in the context below. 
        For complex queries spanning multiple reports, provide a structured summary highlighting key comparisons or trends.
        If certain information is unavailable, state so clearly and suggest alternative approaches to obtain it.

        Now it's your turn!
        
        {question}

        {context}
        """
    
    prompt = ChatPromptTemplate.from_template(template)

    return prompt

def create_embeddings_openai(model="text-embedding-ada-002") -> OpenAIEmbeddings:
    """
    # Create embeddings using OpenAI
    """
    # Initialize the OpenAIEmbeddings class
    embeddings = OpenAIEmbeddings(model=model)

    return embeddings

def create_embeddings_vertexai(model="text-embedding-004") -> VertexAIEmbeddings:
    """
    # Create embeddings using Vertex AI
    # https://python.langchain.com/docs/integrations/text_embedding/google_vertex_ai_palm/
    """

    creds, _ = google.auth.default(quota_project_id=os.environ["PROJECT_ID"])

    # Initialize the VertexAIEmbeddings class
    embeddings = VertexAIEmbeddings(model_name=model, credentials=creds)

    return embeddings

def create_retriever_qdrant(vector_store: QdrantVectorStore) -> BaseRetriever:
    """
    # Create a Qdrant retriever
    # """
    retriever = vector_store.as_retriever()

    return retriever

def create_retriever_vertexai(max_documents: int = 3) -> VertexAISearchRetriever:
    """
    # Create a Vertex AI retriever
    # https://python.langchain.com/docs/integrations/retrievers/google_vertex_ai_search/
    """
    
    retriever = VertexAISearchRetriever(project_id=os.environ["PROJECT_ID"], location_id=os.environ["LOCATION_ID"], data_store_id=os.environ["DATA_STORE_ID"], max_documents=max_documents)

    return retriever

def create_vector_store_qdrant(location: str, collection_name: str, vector_size: int, embeddings: Embeddings, docs: list) -> QdrantVectorStore:
    """"
    # Create a Qdrant vector store
    """                               
            
    # Initialize the Qdrant client
    qdrant_client = QdrantClient(location=location)

    # Create a collection in Qdrant
    qdrant_client.create_collection(collection_name=collection_name, 
                                 vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))

    # Initialize QdrantVectorStore with the Qdrant client
    qdrant_vector_store = QdrantVectorStore(client=qdrant_client, 
                                         collection_name=collection_name, embedding=embeddings)
    
    # Add the docs to the vector store
    qdrant_vector_store.add_documents(docs)
    
    return qdrant_vector_store

def gemini_is_finished_parser(response: LLMResult) -> bool:
    """
    # Create a custom is_finished_parser to capture Gemini generation completion signals
    # https://docs.ragas.io/en/stable/howtos/customizations/customize_models/#google-vertex
    """
    
    is_finished_list = []
    for g in response.flatten():
        resp = g.generations[0][0]

        # Check generation_info first
        if resp.generation_info is not None:
            finish_reason = resp.generation_info.get("finish_reason")
            if finish_reason is not None:
                is_finished_list.append(
                    finish_reason in ["STOP", "MAX_TOKENS"]
                )
                continue

        # Check response_metadata as fallback
        if isinstance(resp, ChatGeneration) and resp.message is not None:
            metadata = resp.message.response_metadata
            if metadata.get("finish_reason"):
                is_finished_list.append(
                    metadata["finish_reason"] in ["STOP", "MAX_TOKENS"]
                )
            elif metadata.get("stop_reason"):
                is_finished_list.append(
                    metadata["stop_reason"] in ["STOP", "MAX_TOKENS"] 
                )

        # If no finish reason found, default to True
        if not is_finished_list:
            is_finished_list.append(True)

    return all(is_finished_list)

def generate_answers_contexts(chain: RunnableSerializable, questions: list) -> Tuple[List, List]:
    """
    # Generate answers from a chain using a list of questions
    """

    answers = []
    contexts = []

    # Loop over the list of questions and call the chain to get the answer and context
    for question in questions:
        print(question)

        # Call the chain to get answers and contexts
        response = chain.invoke({"question" : question})
        print(response)

        # Capture the answer and context 
        answers.append(response["response"].content)
        contexts.append([context.page_content for context in response["context"]])

    return answers, contexts

def get_time_string(format: str = "%Y%m%d%H%M%S") -> str:
    return time.strftime(format)

def process_directory(path: str, glob: str, loader_cls: str = None, use_multithreading: bool = False ) -> None:
    """
    # Process the docs in a directory
    """

    if loader_cls is None:
        loader = DirectoryLoader(path=path, glob=glob, show_progress=True, use_multithreading=use_multithreading)
    else:
        loader = DirectoryLoader(path=path, glob=glob, loader_cls=loader_cls, show_progress=True, use_multithreading=use_multithreading)
    
    docs = loader.load()
    
    return docs

def run_ragas_evaluation(chain: RunnableSerializable, testset_name: str, 
                         eval_metrics: list = [answer_correctness, answer_relevancy, context_recall, context_precision, faithfulness], 
                         use_google: bool = False):
    """
    # Run a Ragas evaluation
    """   

    # Get the questions and groundtruths from the dataframe
    testset_df = pd.read_csv(testset_name)
    
    # Get the questions from testset
    questions = testset_df["user_input"].values.tolist()
    questions = [str(question) for question in questions]

    # Get the groundtruths from testset
    groundtruths = testset_df["reference"].values.tolist()
    groundtruths = [str(ground_truth) for ground_truth in groundtruths]

    # Get answer and context for all the questions
    answers = []
    contexts = []
    answers, contexts = generate_answers_contexts(chain=chain, questions=questions)

    # Create the input dataset 
    input_dataset = Dataset.from_dict({"question" : questions, "answer" : answers, "contexts" : contexts, "ground_truth" : groundtruths})

    # Check if we're using Google or OpenAI 
    if use_google: 
        creds, _ = google.auth.default(quota_project_id=os.environ["PROJECT_ID"])
        
        llm = ChatVertexAI(credentials=creds, model_name=os.environ["GOOGLE_LLM_MODEL_NAME"],)
        embeddings = VertexAIEmbeddings(credentials=creds, model_name=os.environ["GOOGLE_EMBEDDING_MODEL_NAME"])

        llm = LangchainLLMWrapper(llm, is_finished_parser=gemini_is_finished_parser)
        embeddings = LangchainEmbeddingsWrapper(embeddings)

        run_config=RunConfig(max_workers=2)
    else:
        llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

        run_config=RunConfig(max_workers=16)

    # Run the Ragas evaluation using the input dataset and eval metrics
    ragas_results = evaluate(input_dataset, eval_metrics, llm=llm, embeddings=embeddings, run_config=run_config)
      
    ragas_results_df = ragas_results.to_pandas()
    
    return ragas_results, ragas_results_df

##############

if __name__ == "__main__":
    # Load environment variables from .env file
    
    load_dotenv()

    create_embeddings_openai()