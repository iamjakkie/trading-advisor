# agents/strategy_agent.py
from transformers import pipeline

class StrategyAgent:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.qa_pipeline = pipeline("question-answering")

    def ask(self, question):
        result = self.qa_pipeline(question=question, context=self.knowledge_base)
        return result['answer']

# This part can be moved to main.py as well, but keeping it here for clarity.
def create_agent(books_summary, articles_summary):
    knowledge_base = books_summary[0]['summary_text'] + " " + articles_summary[0]['summary_text']
    return StrategyAgent(knowledge_base)