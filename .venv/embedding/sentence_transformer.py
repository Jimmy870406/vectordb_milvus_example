from abstract.abstract_embedding import Embedding_Model
from pymilvus import model

class Sentence_Transformer(Embedding_Model):
    def __init__(self):
        pass

    def embedding(self, resource):
        sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",  # Specify the model name
            device="cpu"  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        )

        docs_embeddings = sentence_transformer_ef(resource)
        return docs_embeddings