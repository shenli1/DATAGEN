from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
class LanguageModelManager:
    def __init__(self):
        """Initialize the language model manager"""
        self.llm = None
        self.power_llm = None
        self.json_llm = None
        self.initialize_llms()

    def initialize_llms(self):
        """Initialize language models"""
        try:
            model_name = "google/gemini-2.5-pro-exp-03-25:free"
            self.llm = ChatOpenAI(model=model_name,base_url="https://openrouter.ai/api/v1", temperature=0, max_tokens=4096)
            self.power_llm = ChatOpenAI(model=model_name,base_url="https://openrouter.ai/api/v1",temperature=0.5, max_tokens=4096)
            self.json_llm = ChatOpenAI(
                model=model_name,
                base_url="https://openrouter.ai/api/v1",
                model_kwargs={"response_format": {"type": "json_object"}},
                temperature=0,
                max_tokens=4096
            )
            print("Language models initialized successfully.")
        except Exception as e:
            print(f"Error initializing language models: {str(e)}")
            raise

    def get_models(self):
        """Return all initialized language models"""
        return {
            "llm": self.llm,
            "power_llm": self.power_llm,
            "json_llm": self.json_llm
        }
    
if __name__ == "__main__":
    load_dotenv()
    lm = LanguageModelManager()
    print(lm.get_models())
    print(os.getenv("OPENAI_API_KEY"))
    print(os.getenv("CONDA_PATH"))
    response = lm.llm.invoke("Hello, world!")
    print(response)
