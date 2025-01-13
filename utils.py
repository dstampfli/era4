
import os
import time
from operator import itemgetter
from typing import List, Tuple

import google.auth
import pandas as pd
from datasets import Dataset
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_community import VertexAISearchRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

##############

# Create a text splitter using recursive character text splitter
# https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
def chunk_docs_recursive(docs: list, 
						 chunk_size=500, 
						 chunk_overlap=50
						 ) -> list:

	text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
												chunk_overlap=chunk_overlap)

	chunks = text_splitter.split_documents(docs)

	return chunks

# Create embeddings using OpenAI
def create_embeddings_openai(model="text-embedding-ada-002") -> OpenAIEmbeddings:

	# Initialize the OpenAIEmbeddings class
	embeddings = OpenAIEmbeddings(model=model)

	return embeddings
\
# Create a Langchain chain using OpenAI
def create_chain_openai (model: str, 
				  prompt_template: ChatPromptTemplate, 
				  retriever: BaseRetriever
				  ) -> RunnableSerializable:

	llm = ChatOpenAI(model=model)
		
	chain = (
		{"context": itemgetter("question") | retriever, "question": itemgetter("question")} 
		| RunnablePassthrough.assign(context=itemgetter("context")) 
		| {"response": prompt_template | llm, "context": itemgetter("context")}
		)

	return chain

# Create a Langchain chain using OpenAI
# https://python.langchain.com/docs/integrations/llms/google_ai/
# https://python.langchain.com/docs/integrations/chat/google_generative_ai/
# https://ai.google.dev/gemini-api/docs/safety-settings 
def create_chain_vertexai (model_name: str, 
						   prompt_template: ChatPromptTemplate, 
						   retriever: BaseRetriever) -> RunnableSerializable:

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
		{"context": itemgetter("question") | retriever, "question": itemgetter("question")} 
		| RunnablePassthrough.assign(context=itemgetter("context")) 
		| {"response": prompt_template | llm, "context": itemgetter("context")}
		)

	return chain

# Create a prompt template
# https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/#chatprompttemplate
# https://python.langchain.com/v0.2/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
def create_chat_prompt_template(template: str = None) -> ChatPromptTemplate:
	
	if template is None:
		template = '''
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
		'''
	
	prompt = ChatPromptTemplate.from_template(template)

	return prompt

# Create embeddings using Vertex AI
# https://python.langchain.com/docs/integrations/text_embedding/google_vertex_ai_palm/
def create_embeddings_vertexai(model="text-embedding-004") -> VertexAIEmbeddings:

	creds, _ = google.auth.default(quota_project_id=os.environ["PROJECT_ID"])

	# Initialize the VertexAIEmbeddings class
	embeddings = VertexAIEmbeddings(model_name=model, 
								 credentials=creds)

	return embeddings

# Create a Qdrant vector store
def create_qdrant_vector_store(location: str, 
							   collection_name: str, 
							   vector_size: int, 
							   embeddings: Embeddings, 
							   docs: list) -> QdrantVectorStore:

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

# Create a Qdrant retriever
def create_retriever_qdrant(vector_store: QdrantVectorStore) -> BaseRetriever:

	retriever = vector_store.as_retriever()

	return retriever

# Create a Vertex AI retriever
# https://python.langchain.com/docs/integrations/retrievers/google_vertex_ai_search/
def create_retriever_vertexai() -> VertexAISearchRetriever:

	retriever = VertexAISearchRetriever(project_id=os.environ["PROJECT_ID"], location_id=os.environ["LOCATION_ID"], data_store_id=os.environ["DATA_STORE_ID"], max_documents=3)

	return retriever

# Generate answers from a chain using a list of questions
def generate_answers_contexts(chain: RunnableSerializable, 
							  questions: list
							  ) -> Tuple[List, List]:
	
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

# Load docs from a directory
def process_directory(
    path: str, 
    glob: str, 
    loader_cls: str = None, 
    use_multithreading: bool = True
) -> None:
    	
	if loader_cls is None:
		loader = DirectoryLoader(path=path, 
							glob=glob, 
						   	show_progress=True, 
						   	use_multithreading=use_multithreading)
	else:
		loader = DirectoryLoader(path=path, 
						   	glob=glob, 
							loader_cls=loader_cls, 
						   	show_progress=True, 
						   	use_multithreading=use_multithreading)
	
	docs = loader.load()
	
	return docs

# Run a Ragas evaluation 
def run_ragas_evaluation(chain: RunnableSerializable, 
						questions: list, 
						groundtruths: list, 
						eval_metrics: list = [answer_correctness, answer_relevancy, context_recall, context_precision, faithfulness]
						):
	
	answers = []
	contexts = []
	answers, contexts = generate_answers_contexts(chain=chain, 
                                               questions=questions)

	# Create the input dataset 
	input_dataset = Dataset.from_dict({"question" : questions,       	# From the dataframe
										"answer" : answers,             # From the chain
										"contexts" : contexts,          # From the chain
										"ground_truth" : groundtruths   # From the dataframe
										})

	# Run the Ragas evaluation using the input dataset and eval metrics
	ragas_results = evaluate(input_dataset, 
                          eval_metrics)
      
	ragas_results_df = ragas_results.to_pandas()
	
	return ragas_results, ragas_results_df

##############

def test_chunk_docs_recursive(): 
	docs = process_directory(path="docs/10k/html", 
						  glob="**/*.html", 
						  loader_cls=None, 
						  use_multithreading=True)
	
	chunks = chunk_docs_recursive(docs=docs)

	print(f"\nNumber of chunks = {len(chunks)}\n")

	print(f"First chunk = {chunks[0].page_content}")

def test_create_chain_qdrant():

	docs = process_directory(path="docs/10k/html", 
						  glob="**/*.html", 
						  loader_cls=None, 
						  use_multithreading=True)

	chunks = chunk_docs_recursive(docs=docs)

	embeddings = create_embeddings_openai()

	vector_store = create_qdrant_vector_store(":memory:", 
										   "holiday-test", 
										   1536, 
										   embeddings, 
										   chunks)

	retriever = create_retriever_qdrant(vector_store)

	chat_prompt_template = create_chat_prompt_template()

	chain = create_chain("gpt-4o-mini", 
					  chat_prompt_template, 
					  retriever)
	
	result = chain.invoke({"question" : "What is the annual revenue of Uber?"})
	
	print(result)

def test_create_chain_vertexai():
	retreiver = create_retriever_vertexai()
	chat_prompt_template = create_chat_prompt_template()
	chain = create_chain('gemini-1.5-flash', chat_prompt_template, retreiver)
	result = chain.invoke({'question' : 'What is the annual revenue of Uber?'})
	print(result)

def test_create_chat_prompt_template():
	prompt = create_chat_prompt_template()
	
	print(prompt)

def test_create_embeddings_openai():
	text = "What is the annual revenue of Uber?"
	
	embeddings = create_embeddings_openai()
	vector = embeddings.embed_query(text)
	
	print(vector)

def test_create_embeddings_vertexai():
	text = 'What is the annual revenue of Uber?'
	embeddings = create_embeddings_vertexai()
	vector = embeddings.embed_query(text)
	print(vector)
	return embeddings

def test_create_qdrant_vector_store():
	docs = process_directory(path="docs/10k/html", 
						  glob="**/*.html", 
						  loader_cls=None, 
						  use_multithreading=True)
	
	chunks = chunk_docs_recursive(docs=docs)
	
	embeddings = create_embeddings_openai()

	vector_store = create_qdrant_vector_store(":memory:", 
										   "holiday-test", 
										   1536, 
										   embeddings, 
										   chunks)
	
	print(vector_store.collection_name)

def test_create_retriever_qdrant(text):
	docs = process_directory(path="docs/10k/html", 
						  glob="**/*.html", 
						  loader_cls=None, 
						  use_multithreading=True)
	
	chunks = chunk_docs_recursive(docs=docs)
	
	embeddings = create_embeddings_openai()

	vector_store = create_qdrant_vector_store(":memory:", 
										   "holiday-test", 
										   1536, 
										   embeddings, 
										   chunks)
	
	retriever = create_retriever_qdrant(vector_store)
	
	docs = retriever.invoke(text)

	print(docs[0])

def test_create_retriever_vertexai(text:str):
	retriever = create_retriever_vertexai()
	docs = retriever.invoke(text)
	print(docs[0])

def test_generate_answers_contexts():
	docs = process_directory(path="docs/10k/html", 
						  glob="**/*.html", 
						  loader_cls=None, 
						  use_multithreading=True)
	
	chunks = chunk_docs_recursive(docs=docs)

	embeddings = create_embeddings_openai()

	vector_store = create_qdrant_vector_store(":memory:", 
										   "holiday-test", 
										   1536, 
										   embeddings, 
										   chunks)
	
	retriever = create_retriever_qdrant(vector_store)

	chat_prompt_template = create_chat_prompt_template()

	chain = create_chain("gpt-4o-mini", 
					  chat_prompt_template, 
					  retriever)
	
	questions = ["What is the annual revenue of Lyft?",
			  "What is the annual revenue of Uber?",
			  "Which company has a larger annual revenue - Lyft or Uber?"]	
	
	answers, contexts = generate_answers_contexts(chain=chain, 
											   questions=questions)
	
	print(f"Total number of answers = {len(answers)}")
	print(f"Total number of contexts = {len(contexts)}")

def test_process_directory():
	docs = process_directory(path="docs/10k/html", 
						  glob="**/*.html", 
						  loader_cls=None, 
						  use_multithreading=True)
	print(len(docs))

def test_run_ragas_evaluation():
    docs = process_directory(
        path="docs/10k/html",
        glob="**/*.html",
        loader_cls=None,
        use_multithreading=True
    )

    chunks = chunk_docs_recursive(docs=docs)

    embeddings = create_embeddings_openai()

    vector_store = create_qdrant_vector_store(
        ":memory:",
        "holiday-test",
        1536,
        embeddings,
        chunks
    )

    retriever = create_retriever_qdrant(vector_store)

    chat_prompt_template = create_chat_prompt_template()

    chain = create_chain("gpt-4o-mini", chat_prompt_template, retriever)

    testset_df = pd.read_csv("testsets/10k_testset.csv")

    questions = testset_df["user_input"].values.tolist()
    questions = [str(question) for question in questions]
    top_5_questions = questions[:5]

    groundtruths = testset_df["reference"].values.tolist()
    groundtruths = [str(ground_truth) for ground_truth in groundtruths]
    top_5_groundtruths = groundtruths[:5]

    eval_metrics = [
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness
    ]
    
    ragas_results, ragas_results_df = run_ragas_evaluation(chain, 
                                                           	top_5_questions, 
                                                        	top_5_groundtruths, 
                                                        	eval_metrics)

    timestr = time.strftime("%Y%m%d%H%M%S")
    ragas_results_df.to_csv(f"evaluations/10x_test_testset_evaluation_{timestr}.csv")

    print(ragas_results)

##############

if __name__ == "__main__":
	# Load environment variables from .env file
	from dotenv import load_dotenv
	load_dotenv()

	# Run the tests
	test_process_directory()
	test_create_embeddings_openai()
	test_chunk_docs_recursive()
	test_create_qdrant_vector_store()
	test_create_retriever_qdrant("What is the annual revenue for Uber?")
	test_create_chat_prompt_template()
	test_create_chain_qdrant()
	
	test_generate_answers_contexts()
	test_run_ragas_evaluation()

	test_create_embeddings_vertexai()
	test_create_retriever_vertexai("What is the annual revenue of Uber?")
	test_create_chain_vertexai()
