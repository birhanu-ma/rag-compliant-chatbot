from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class ComplaintRAGChain:
    def __init__(self, llm, vector_db):
        """
        Optimized Senior Analyst RAG Pipeline.
        """
        self.llm = llm
        # SPEED BOOST: Reduced k from 5 to 3. 
        # Fewer documents = much faster CPU processing.
        self.retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        
        # We removed the word "Answer:" or "Response:" from the end.
        # By ending with the bullet point symbol, the model will just continue from there.
        self.template = """<|user|>
                        You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.
                        
                        Context: {context}
                        
                        Question: {question}<|end|>
                        <|assistant|>
                        * """

        self.prompt = PromptTemplate.from_template(self.template)
        # Standard LCEL Chain setup
        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        """Formats retrieved documents. Limits text to prevent CPU lag."""
        # We only take the first 1000 characters per doc to keep the prompt slim
        return "\n\n".join(doc.page_content[:1000] for doc in docs)

    def query(self, user_input):
        """
        Executes the RAG query and returns result + sources.
        """
        # 1. Fetch relevant documents
        source_docs = self.retriever.invoke(user_input)
        
        # 2. Format documents for the prompt
        context_text = self._format_docs(source_docs)
        
        # 3. Invoke the chain logic
        # This uses the shorter prompt and limited context for speed
        llm_chain = self.prompt | self.llm | StrOutputParser()
        response_text = llm_chain.invoke({"context": context_text, "question": user_input})
        
        return {
            "result": response_text,
            "source_documents": source_docs
        }