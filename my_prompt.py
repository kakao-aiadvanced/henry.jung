from langchain import hub
from langchain.prompts import PromptTemplate

# prompt = hub.pull("rlm/rag-prompt")
retrieval_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following context to answer the question.
If you don't know the answer, just say that you don't know.

Context: {context}
Question: {question}
Answer:"""
)

# retrieval_grader_prompt = hub.pull("efriis/self-rag-retrieval-grader")
retrieval_grader_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a grader assessing the relevance of a retrieved document to a user question. If the document is relevant, return True; otherwise, return False.
    
Context: {context}
Question: {question}
Answer:"""
)

result_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the document to generate a summarized answer,  and include the document as a reference at the end.

Context: {context}
Question: {question}
Answer:"""
)