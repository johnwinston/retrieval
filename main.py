from config import Config
from interface import Interface
import json
from tqdm import tqdm


def get_user_confirmation(prompt):
    return input(prompt).lower().startswith('y')



def main():
    config = Config()
    interface = Interface()

    if get_user_confirmation("Generate new embeddings?"):
        for entry in tqdm(config.DESCRIPTIONS):
            config.EMBEDDINGS.append(
                    interface.get_embedding(
                        config.DESCRIPTIONS[entry]['description']
                        )
                    )
        with open(config.EMBEDDINGS_FILE, 'w') as f:
            json.dump(config.EMBEDDINGS, f)

    print(config.EMBEDDINGS[0])

if __name__ == "__main__":
    main()
