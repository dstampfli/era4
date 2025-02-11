# Run all of the tests - python3 -m unittest test_utils.py  
# Run a specific test - python -m unittest test_utils.TestUtils.test_process_directory

import os
import unittest

from dotenv import load_dotenv
import magic

from .utils import *

##############

class TestUtils(unittest.TestCase):

    def test_chunk_docs_recursive(self):
        print("Running test_chunk_docs_recursive")
        # Arrange
        docs = process_directory(path="docs/10k/html", glob="**/*.html", loader_cls=None, use_multithreading=False,)
        # Act
        chunks = chunk_docs_recursive(docs=docs)
        # Assert
        self.assertGreater(len(chunks), 0, "Chunks should not be empty")
        self.assertIsNotNone(chunks[0].page_content, "First chunk should have content")

    def test_create_chain_openai(self):
        print("Running test_create_chain_qdrant")
        # Arrange
        load_dotenv()
        docs = process_directory(path="docs/10k/html", glob="**/*.html", loader_cls=None, use_multithreading=False,)
        chunks = chunk_docs_recursive(docs=docs)
        embeddings = create_embeddings_openai()
        vector_store = create_vector_store_qdrant(location=":memory:", collection_name="unittest", vector_size=1536, embeddings=embeddings, docs=chunks,)
        retriever = create_retriever_qdrant(vector_store)
        chat_prompt_template = create_chat_prompt_template()
        chain = create_chain_openai(model="gpt-4o-mini", prompt_template=chat_prompt_template, retriever=retriever,)        
        # Act 
        answer = chain.invoke({"question" : "What was Uber's revenue in 2023?"})
        print(answer)
        # Assert
        self.assertIsNotNone(answer, "Answer should not be None")

    def test_create_chain_vertexai(self):
        print("Running test_create_chain_vertexai")
        # Arrange
        load_dotenv()
        retreiver = create_retriever_vertexai()
        chat_prompt_template = create_chat_prompt_template()
        chain = create_chain_vertexai(model_name="gemini-1.5-flash", prompt_template=chat_prompt_template, retriever=retreiver)
        # Act
        answer = chain.invoke({"question" : "What was Uber's revenue in 2023?"})
        print(answer)
        # Assert
        self.assertIsNotNone(answer, "Answer should not be None")

    def test_create_chat_prompt_template(self):
        print("Running test_create_chat_prompt_template")
	    # Arrange
        # Act
        prompt = create_chat_prompt_template()
        # Assert
        self.assertIsNotNone(prompt, "Prompt should not be None")

    def test_create_embeddings_openai(self):
        print("Running test_create_embeddings_openai")
        # Arrange
        load_dotenv()
        embeddings = create_embeddings_openai()
        # Act
        vector = embeddings.embed_query("What was Uber's revenue in 2023?")
        print(vector)
        # Assert
        self.assertIsNotNone(vector, "Vector should not be None")

    def test_create_embeddings_vertexai(self):
        print("Running test_create_embeddings_vertexai")
        # Arrange
        load_dotenv()
        embeddings = create_embeddings_vertexai()
        # Act
        vector = embeddings.embed_query("What was Uber's revenue in 2023?")
        print(vector)
        # Assert
        self.assertIsNotNone(vector, "Vector should not be None")

    def test_create_retriever_qdrant(self):
        print("Running test_create_retriever_qdrant")
        # Arrange
        load_dotenv()
        docs = process_directory(path="docs/test", glob="**/*.md", loader_cls=None, use_multithreading=False,)
        chunks = chunk_docs_recursive(docs=docs)
        embeddings = create_embeddings_openai()
        vector_store = create_vector_store_qdrant(location=":memory:", collection_name="unittest", vector_size=1536, embeddings=embeddings, docs=chunks,)
        retriever = create_retriever_qdrant(vector_store)
        # Act
        docs = retriever.invoke("What was Uber's revenue in 2023?")
        print(f"Found {len(docs)} docs")
        # Assert
        self.assertGreater(len(docs), 0, "Docs should not be empty")
    
    def test_create_retriever_vertexai(self):
        print("Running test_create_retriever_vertexai")
        # Arrange
        load_dotenv()
        retriever = create_retriever_vertexai()
        # Act
        docs = retriever.invoke("What was Uber's revenue in 2023?")
        print(f"Found {len(docs)} docs")
        # Assert
        self.assertGreater(len(docs), 0, "Docs should not be empty")

    def test_create_vector_store_qdrant(self):
        print("Running test_create_vector_store_qdrant")
        # Arrange
        load_dotenv()
        docs = process_directory(path="docs/10k/html", glob="**/*.html", loader_cls=None, use_multithreading=False,)
        chunks = chunk_docs_recursive(docs=docs)
        embeddings = create_embeddings_openai()
        # Act   
        vector_store = create_vector_store_qdrant(location=":memory:", collection_name="unittest", vector_size=1536, embeddings=embeddings, docs=chunks,)
        # Assert
        self.assertIsNotNone(vector_store, "Vector store should not be None",)

    def test_generate_answers_contexts(self):
        print("Running test_generate_answers_contexts")
        # Arrange
        load_dotenv()
        docs = process_directory(path="docs/10k/html", glob="**/*.html", loader_cls=None, use_multithreading=True,)
        chunks = chunk_docs_recursive(docs=docs)
        embeddings = create_embeddings_openai()
        vector_store = create_vector_store_qdrant(location=":memory:", collection_name="unittest", vector_size=1536, embeddings=embeddings, docs=chunks,)
        retriever = create_retriever_qdrant(vector_store)
        chat_prompt_template = create_chat_prompt_template()
        chain = create_chain_openai(model="gpt-4o-mini", prompt_template=chat_prompt_template, retriever=retriever,) 
        questions = ["What was Lyft's revenue in 2013?", "What was Uber's revenue in 2013?",]    
        # Act
        answers, contexts = generate_answers_contexts(chain=chain, questions=questions,)
        # Assert
        self.assertGreater(len(answers), 0, "Answers should not be empty")
        self.assertGreater(len(contexts), 0, "Contexts should not be empty")
    
    def test_get_time_string(self):
        print("Running test_get_time_string")
        # Arrange
        # Act
        timestr = get_time_string()
        # Assert
        self.assertIsNotNone(timestr, "Timestr should not be None")

    def test_process_directory(self):
        print("Running test_process_directory")
	    # Arrange
        # Act
        docs = process_directory(path="docs/10k/html", glob="**/*.html", loader_cls=None, use_multithreading=False,)
        # Assert
        self.assertGreater(len(docs), 0, "Docs should not be empty")

    def test_run_ragas_evaluation(self):
        print("Running test_run_ragas_evaluation")
        # Arrange
        load_dotenv()
        docs = process_directory(path="docs/10k/html", glob="**/*.html", loader_cls=None, use_multithreading=True,)
        chunks = chunk_docs_recursive(docs=docs)
        embeddings = create_embeddings_openai()
        vector_store = create_vector_store_qdrant(location=":memory:", collection_name="unittest", vector_size=1536, embeddings=embeddings, docs=chunks,)
        retriever = create_retriever_qdrant(vector_store)
        chat_prompt_template = create_chat_prompt_template()
        chain = create_chain_openai(model="gpt-4o-mini", prompt_template=chat_prompt_template, retriever=retriever,)
        eval_metrics = [answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness,]
        # Act
        ragas_results, ragas_results_df = run_ragas_evaluation(chain=chain, testset_name="testsets/unittest_testset-gpt-4o-mini.csv", eval_metrics=eval_metrics,)
        timestr = time.strftime("%Y%m%d%H%M%S")
        ragas_results_df.to_csv(f"evaluations/unittest_testset_evaluation_{timestr}.csv")
        # Assert
        self.assertIsNotNone(ragas_results, "Ragas results should not be None")

