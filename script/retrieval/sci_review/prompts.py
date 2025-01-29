from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain import hub
from pydantic import BaseModel, Field

grade_doc_prompt = PromptTemplate(
    template=
"""You are a grader assessing relevance of a retrieved document to a user question.

Here is the retrieved document:

{context}

Here is the user question:

{question}

If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
    input_variables=["context", "question"],
)

class grade_doc_data(BaseModel):
    """Binary score for relevance check."""

    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


rag_prompt:PromptTemplate = hub.pull("rlm/rag-prompt")