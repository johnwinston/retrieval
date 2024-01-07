from config import Config
from interface import Interface
import json
from tqdm import tqdm

def get_user_confirmation(prompt):
    return input(prompt).lower().startswith('y')

def cosine_similarity(vec1, vec2):
    import numpy as np
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

    idx = vector_set.index(most_similar)
    print(idx)
    return idx

def main():
    config = Config()
    interface = Interface()

    if get_user_confirmation("Generate new embeddings? [y/n]\n"):
        config.EMBEDDINGS = []
        for entry in tqdm(config.DESCRIPTIONS):
            config.EMBEDDINGS.append(
                    interface.get_embedding(
                        config.DESCRIPTIONS[entry]['description']
                        )
                    )
        with open(config.EMBEDDINGS_FILE, 'w') as f:
            json.dump(config.EMBEDDINGS, f)

    print(config.DESCRIPTIONS)
    if get_user_confirmation("Generate new descriptions? [y/n]\n"):
        for entry in tqdm(config.DESCRIPTIONS):
            config.DESCRIPTIONS[entry]['description'] =\
                    interface.get_description(
                        config.DESCRIPTIONS[entry]['description']
                        )
        with open(config.DESCRIPTIONS_FILE, 'w') as f:
            json.dump(config.DESCRIPTIONS, f)

    if get_user_confirmation("Generate new descriptions? [y/n]\n"):
        for i, query in tqdm(enumerate(config.SPIDER_QUERY_EMBEDDINGS)):
            most_similar_idx =\
                most_similar_vector(
                        interface.get_embedding(query['question']),
                        config.EMBEDDINGS
                        )
            key = list(config.DESCRIPTIONS)[most_similar_idx]
            original_idx = list(config.DESCRIPTIONS).index(query['db_id'])
            most_similar_dataset = json.dumps({key : config.DESCRIPTIONS[key]})
            dataset = json.dumps({query['db_id'] : config.DESCRIPTIONS[query['db_id']]})
            if dataset == most_similar_dataset:
                continue
            print(query['question'])

            while query['db_id'] != key:
                print(query['db_id'], key)
                orig_desc, ret_desc =\
                        interface.get_descriptions(
                            dataset,
                            most_similar_dataset,
                            query['question']
                            )
                print(config.DESCRIPTIONS[query['db_id']]['description'])
                print(config.DESCRIPTIONS[key]['description'])
                config.DESCRIPTIONS[query['db_id']]['description'] = orig_desc
                config.DESCRIPTIONS[key]['description'] = ret_desc
                print("---------------------")
                print(config.DESCRIPTIONS[query['db_id']]['description'])
                print(config.DESCRIPTIONS[key]['description'])
                print("\n\n")
                orig_emb = config.EMBEDDINGS[original_idx]
                ret_emb = config.EMBEDDINGS[most_similar_idx]
                config.EMBEDDINGS[original_idx] =\
                        interface.get_embedding(
                            config.DESCRIPTIONS[query['db_id']]['description']
                            )
                config.EMBEDDINGS[most_similar_idx] =\
                        interface.get_embedding(
                            config.DESCRIPTIONS[key]['description']
                            )
                if orig_emb == config.EMBEDDINGS[original_idx]:
                    print("Original embedding unchanged")
                if ret_emb == config.EMBEDDINGS[most_similar_idx]:
                    print("Retrieved embedding unchanged")

                most_similar_idx =\
                    most_similar_vector(
                            interface.get_embedding(query['question']),
                            config.EMBEDDINGS
                            )
                key = list(config.DESCRIPTIONS)[most_similar_idx]
                most_similar_dataset = json.dumps({key : config.DESCRIPTIONS[key]})
                dataset = json.dumps({query['db_id'] : config.DESCRIPTIONS[query['db_id']]})
            interface.reset_messages()
            print("Matched!")


if __name__ == "__main__":
    main()
