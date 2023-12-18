from openai import OpenAI
import os

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    return dot_product / (norm_vec1 * norm_vec2)

def most_similar_vector(target_vec, vector_set):
    max_similarity = -1
    most_similar = None

    for vec in vector_set:
        similarity = cosine_similarity(target_vec, vec)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar = vec

    return most_similar, max_similarity

class Interface:
    def __init__(self):
        self.client = OpenAI()
        self.prompt = Prompts()

    def query_chatGPT(self, message):
        try:
            response = self.client.chat.completions.create(
                messages=message,
                model="gpt-4"
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

    def get_descriptions(self,
                        dataset,
                        retrieved_dataset,
                        query
                        ):
        prompt = self.prompt.generate_message(
                    dataset,
                    retrieved_dataset,
                    query
                    )
        response = self.query_chatGPT(prompt)
        print(response)
        orig_desc = response[
                response.find("Original:")+len("Original:")
                :response.find("Retrieved:")
                ]
        ret_desc = response[response.find("Retrieved:")+len("Retrieved:"):]
        return orig_desc.strip(), ret_desc.strip()

class Prompts:
    def __init__(self):
        self.prompt_template = (
            "Original Dataset:\n{}\n\n"
            "Retrieved Dataset:\n{}\n\n"
            "Query:\n{}\n\n"
            "Requests:\n{}\n\n"
            "Descriptions:\n"
        )
        self.request_template = (
            "We retrieve a dataset based on an embedded user query.\n"
            "Generate a better description for the original dataset and retrieved dataset so that it will be retrieved correctly based on the query.\n"
            "Alter the undesired retrieved description so that it will not be retrieved based on the query.\n"
            "Only return the descriptions.\n"
            "Do not include quotation marks.\n"
            "Format the response like this [Original: description\nRetrieved: description]\n"
            "IT IS ORIGINAL AND RETRIEVED ONLY IN THE FORMAT. FORMAT IT RIGHT.\n"
            "Do not include a revised retrieved section, only include Original and Retrieved.\n\n"
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

    def reset_messages(self):
        self.messages = [
                {
                    "role" : "system",
                    "content" : "You produce descriptions."
                }
            ]

    def format_prompt(self,
                      dataset,
                      retrieved_dataset,
                      query):
        return self.prompt_template.format(
                dataset,
                retrieved_dataset,
                query,
                self.request_template
                )
    def format_user_content(self,
                            dataset,
                            retrieved_dataset,
                            query):
        content = self.user_content.copy()
        content["content"] = self.user_content["content"].format(
                self.format_prompt(
                    dataset,
                    retrieved_dataset,
                    query
                    )
                )
        return content
    def generate_message(self,
                         dataset,
                         retrieved_dataset,
                         query):
        self.messages.append(
            self.format_user_content(
                dataset,
                retrieved_dataset,
                query
                ))
        return self.messages
