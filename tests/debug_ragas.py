"""Quick test: single RAGAS call with gpt-4o-mini."""
import os

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI  # noqa: E402

from rag_bench.evaluator import run_ragas_evaluation  # noqa: E402

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

data = [{
    "user_input": "Thu do Viet Nam?",
    "retrieved_contexts": ["Ha Noi la thu do Viet Nam."],
    "response": "Ha Noi.",
    "reference": "Ha Noi.",
}]

result = run_ragas_evaluation(data, model="gpt-4o-mini", client=client)
print("RESULT:", result)
