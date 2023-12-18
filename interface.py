from openai import OpenAI
import os

class Interface:
    def __init__(self):
        self.client = OpenAI()
        self.prompt = Prompts()

    def query_chatGPT(self, message):
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=message
                )
            response = response.choices[0].message.content
        except Exception as e:
            print(e)
            response = ""
        return response

    def get_embedding(self, text):
        return self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text,
                    encoding_format="float"
                    ).data[0].embedding

class Prompts:
    def __init__(self):
        self.prompt_template = (
            "Dataset:\n{}\n\n"
            "Description:\n{}\n\n"
            "Retrieved dataset:\n{}\n\n"
            "Retrieved dataset description:\n"
        )
        self.request_template = (
            "We will retrieve a dataset based on an embedded user query.\n"
            "Generate a better description for the retrieved dataset and given dataset if it fails.\n\n"
        )
        self.messages = [
                {
                    "role" : "system",
                    "content" : "You produce descriptions."
                }
            ]
        self.user_content = {
                "role" : "user",
                "content" : "{}"
                }

    def format_prompt(self, dataset, description, retrieved_dataset):
        return self.prompt_template.format(
                dataset,
                description,
                retrieved_dataset
                )
    def format_user_content(self, dataset, description, retrieved_dataset):
        return self.user_content.format(
                self.format_prompt(
                    dataset,
                    description,
                    retrieved_dataset
                    )
                )

    def generate_message(self, dataset, description, retrieved_dataset):
        self.messages.append(self.format_user_content(
            dataset,
            description,
            retrieved_dataset
            ))
        return self.messages
