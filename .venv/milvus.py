from pymilvus import (connections, utility, model, CollectionSchema, Collection)
from utility.config_provider import Config_Provider

class Milvus:
    def __init__(self):
        pass

    def connect(self):
        connections.connect("default",
                            uri=Config_Provider().milvus_uri,
                            token=Config_Provider().token)
        print(f"Connecting to Milvus: Success")

    def list_collections(self):
        print(utility.list_collections())

    def has_collections(self, name):
        return utility.has_collection(name)

    def create_collection(self, name, field, description):
        schema = CollectionSchema(fields=field, description=description)
        collection = Collection(name, schema, consistency_level="Strong", properties={"collection.ttl.seconds": 15})
        print("\ncollection created:", name)
        return collection

    def drop_collection(self, collection_name):
        if self.has_collections(collection_name):
            collection = Collection(name=collection_name)
            collection.drop()
            print(f"Collection '{collection_name}' has been dropped.")

    def create_index(self, collection_name, field_name, index_type, metric_type, params):
        index = {"index_type": index_type, "metric_type": metric_type, "params": params}
        collection = Collection(name=collection_name)
        collection.create_index(field_name, index)

    def insert(self, collection_name, entity):
        collection = Collection(name=collection_name)
        insert_result = collection.insert(entity)
        return insert_result

    def embed(self, doc, embedding_model):
        return embedding_model.embedding(doc)

    def search(self, collection_name, search_vectors, search_field, search_params):
        collection = Collection(name=collection_name)
        collection.load()
        result = collection.search(search_vectors, search_field, search_params, limit=3, output_fields=["source"])
        collection.release()
        return result

    def drop_all(self, collection_name, expectation):
        collection = Collection(name=collection_name)
        collection.delete(expectation)
        index_info = collection.indexes
        index_info.clear()
        collection.drop()