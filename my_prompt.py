from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

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

agent_router_system_prompt = """You are an expert at routing a user question to a vectorstore or web search.
Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks.
You do not need to be stringent with the keywords in the question related to these topics.
Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
Return the a JSON with a single key 'datasource' and no premable or explanation. Question to route"""

agent_router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_router_system_prompt),
        ("human", "question: {question}"),
    ]
)

agent_retrieval_grader_system_prompt = """You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """

agent_retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_retrieval_grader_system_prompt),
        ("human", "question: {question}\n\n document: {document} "),
    ]
)


agent_generate_system_prompt = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise"""

agent_generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_generate_system_prompt),
        ("human", "question: {question}\n\n context: {context} "),
    ]
)


agent_hallucination_grader_system_prompt = """You are identify any hallucinations in the answer—statements, claims, or details that are not supported by or grounded in the provided documents. 
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

agent_hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_hallucination_grader_system_prompt),
        ("human", "documents: {documents}\n\n answer: {generation} "),
    ]
)


agent_response_grader_system_prompt = """You are a grader assessing whether an
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

agent_response_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_response_grader_system_prompt),
        ("human", "question: {question}\n\n answer: {generation} "),
    ]
)