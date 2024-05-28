from abstract.abstract_embedding import Embedding_Model
from pymilvus import model
from utility.config_provider import Config_Provider

class Openai_Embedding_Model(Embedding_Model):
    def __init__(self):
        pass

    def embedding(self, resource):
        openai_ef = model.dense.OpenAIEmbeddingFunction(
            model_name='text-embedding-3-large',  # Specify the model name
            dimensions=512,  # Set the embedding dimensionality according to MRL feature.
            api_key = Config_Provider().openai_api_key
        )

        docs_embeddings = openai_ef(resource)
        return docs_embeddings