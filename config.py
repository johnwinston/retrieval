from interface import Interface

class Config:
    def __init__(self):
        self.SPIDER_SCHEMA_FILE = './data/spider_schema.json'
        self.DESCRIPTIONS_FILE = './data/descriptions.json'
        self.NO_DESCRIPTIONS_FILE = './data/no_descriptions.json'
        self.EMBEDDINGS_FILE = './data/embeddings.json'

        self.SPIDER_SCHEMA = self.inport(self.SPIDER_SCHEMA_FILE)
        self.DESCRIPTIONS = self.inport(self.DESCRIPTIONS_FILE)
        self.NO_DESCRIPTIONS = self.inport(self.NO_DESCRIPTIONS_FILE)
        self.EMBEDDINGS = self.inport(self.EMBEDDINGS_FILE)
    
    def inport(self, file):
        import os
        if not os.path.exists(file):
            print('File not found: {}'.format(file))
            return []

        with open(file, 'r') as f:
            import json
            data = json.load(f)
            return data
