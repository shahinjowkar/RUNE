from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a AI tutor."),
    ("human", "Explain this topic clearly: {topic}")
])

model = ChatOllama(
    model="llama3",
    temperature=0.2,
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "RAG"})
print(result)
