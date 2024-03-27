from mistral import Mistral7B
from image import stub


@stub.local_entrypoint()
def main():
    model = Mistral7B()
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
        "What is the fable involving a fox and grapes?",
        "What were the major contributing factors to the fall of the Roman Empire?",
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "What is the product of 9 and 8?",
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
    ]
    
    model.generate.remote(questions)
    
    # for question in questions:
    #     print("Sending new request:", question, "\n\n")
    #     for text in model.completion_stream.remote_gen(question):
    #         print(text, end="", flush=True)