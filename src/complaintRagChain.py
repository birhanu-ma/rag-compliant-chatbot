from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class ComplaintRAGChain:
    def __init__(self, llm, vector_db):
        """
        Senior Analyst RAG Pipeline.
        Uses langchain_core to avoid ModuleNotFoundErrors.
        """
        self.llm = llm
        self.retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        
        # Define the prompt using the core package
        self.template = """
You are a senior analyst at CrediTrust Financial. Use the context provided below to answer the query accurately. 

Context: {context}

Query: {question}

Answer:"""
        
        self.prompt = PromptTemplate.from_template(self.template)
        
        # Build the chain manually without 'langchain.chains'
        # This pipes the data: Context + Question -> Prompt -> LLM -> Text output
        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        """Formats retrieved documents into a clean string for the LLM."""
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, user_input):
        # 1. Fetch documents once
        source_docs = self.retriever.invoke(user_input)
        
        # 2. Format them
        context_text = self._format_docs(source_docs)
        
        # 3. Feed the manual context directly to the prompt/LLM logic
        # We define a sub-chain here that bypasses the retriever
        llm_chain = self.prompt | self.llm | StrOutputParser()
        response_text = llm_chain.invoke({"context": context_text, "question": user_input})
        
        return {
            "result": response_text,
            "source_documents": source_docs
        }