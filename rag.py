import os
import bs4

from langchain.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_openai import ChatOpenAI

import my_prompt
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

api_key = os.environ["OPEN_AI_KEY"]

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = api_key

from langchain.globals import set_verbose
set_verbose(True)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

question = "프롬프트 엔지니어링에 대한 설명"

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=200, chunk_overlap=0)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

### Check
# for split in splits:
#     print(split.page_content)

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

### Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

### MultiQueryRetriever 이것도 사실상 prompt를 함수화한 것 뿐.
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=llm
)

### 여기서 invoke를 할 때 주의할점이 retriever llm을 통해서 하므로 query도 만들어서 실행됨. 그러니까 retriever을 해서 그
### 문서 기준으로 이제 내가 실행할 query를 생성하는게 아니라, retriever을 하기위한 query를 생성하고 그 query를 실행해서
### retriever을 함.
unique_docs = retriever_from_llm.invoke(question)
# len(unique_docs)

### Prompt
# template = '''Answer the question based only on the following context:
# {context}
#
# Question: {question}
# '''
#
# prompt = ChatPromptTemplate.from_template(template)

### Chain
chain = (
        {'context': RunnableLambda(lambda _: format_docs(unique_docs)), 'question': RunnablePassthrough()}
        | my_prompt.retrieval_grader_prompt
        | llm
        | StrOutputParser()
)

### ㄱㄱ
response = chain.invoke(question)

if response == "True":
    chain = (
            {'context': RunnableLambda(lambda _: format_docs(unique_docs)), 'question': RunnablePassthrough()}
            | my_prompt.result_prompt
            | llm
            | StrOutputParser()
    )

    print(chain.invoke(question))
else:
    print("Somthing went wrong.")
