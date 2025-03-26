import os
import logging
from pprint import pprint
from typing import List
from typing_extensions import TypedDict

from langchain.globals import set_verbose
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langgraph.graph import END, StateGraph
from tavily import TavilyClient

from my_prompt import agent_router_prompt, agent_retrieval_grader_prompt, agent_generate_prompt, \
    agent_hallucination_grader_prompt, agent_response_grader_prompt

# Logging & env setup
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

api_key = os.environ["OPEN_AI_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = api_key

set_verbose(True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Web search test
response = tavily.search(query="Where does Messi play right now?", max_results=3)
context = [{"url": r["url"], "content": r["content"]} for r in response["results"]]
response_context = tavily.get_search_context(query="Where does Messi play right now?", search_depth="advanced", max_tokens=500)
response_qna = tavily.qna_search(query="Where does Messi play right now?")

# Vector DB setup
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

# vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# vectorstore.from_documents(documents=doc_splits, collection_name="rag-chroma")
vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=embeddings)

retriever = vectorstore.as_retriever()

# Router
question_router = agent_router_prompt | llm | JsonOutputParser()

# Document relevance grader
retrieval_grader = agent_retrieval_grader_prompt | llm | JsonOutputParser()

# RAG QA chain
rag_chain = agent_generate_prompt | llm | StrOutputParser()

# Hallucination grader
hallucination_grader = agent_hallucination_grader_prompt | llm | JsonOutputParser()

# Answer quality grader
answer_grader = agent_response_grader_prompt | llm | JsonOutputParser()

# Graph state
class GraphState(TypedDict):
    question: str
    generation: str
    websearch: str
    documents: List[str]

# Graph nodes
def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    filtered_docs, websearch = [], "No"

    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score["score"].lower() == "yes":
            filtered_docs.append(d)
        else:
            websearch = "Yes"
    return {"documents": filtered_docs, "question": question, "websearch": websearch}

def web_search(state):
    question = state["question"]
    documents = state.get("documents", [])
    docs = tavily.search(query=question)["results"]
    web_results = Document(page_content="\n".join([d["content"] for d in docs]))
    documents.append(web_results)
    return {"documents": documents, "question": question}

# Edge logic
def route_question(state):
    source = question_router.invoke({"question": state["question"]})
    return source["datasource"]

retry_websearch = False
def decide_to_generate(state):
    global retry_websearch
    if retry_websearch:
        print("failed: not relevant")
        exit(1)
    else:
        retry_websearch = True

    return "websearch" if state["websearch"] == "Yes" else "generate"

regen = False
def grade_generation_v_documents_and_question(state):
    global regen
    docs = "\n\n".join(doc.page_content for doc in state["documents"])

    if hallucination_grader.invoke({"documents": docs, "generation": state["generation"]})["score"] == "yes":
        if regen:
            print("failed: hallucination")
            exit(1)
        regen = True
        return "not useful"
    else:
        return "useful"

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.set_conditional_entry_point(route_question, {
    "web_search": "web_search",
    "vectorstore": "retrieve"
})

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {
    "websearch": "web_search",
    "generate": "generate",
})
workflow.add_edge("web_search", "grade_documents")
workflow.add_conditional_edges("generate", grade_generation_v_documents_and_question, {
    "not supported": "generate",
    "useful": END,
    "not useful": "generate"
})

app = workflow.compile()

# Test
inputs = {"question": "What is attack?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])

# import streamlit as st
#
# def main():
#     # ----------------------------------------------------------------------
#     # Streamlit 앱 UI
#     st.title("Research Assistant powered by OpenAI")
#
#     input_topic = st.text_input(
#         ":female-scientist: Enter a topic",
#         value="What is adversarial attack??",
#     )
#
#     generate_report = st.button("Generate Report")
#
#     if generate_report:
#         with st.spinner("Generating Report"):
#             inputs = {"question": input_topic}
#             for output in app.stream(inputs):
#                 for key, value in output.items():
#                     print(f"Finished running: {key}:")
#             final_report = value["generation"]
#             st.markdown(final_report)
#
#     st.sidebar.markdown("---")
#     if st.sidebar.button("Restart"):
#         st.session_state.clear()
#         st.experimental_rerun()
#
# main()