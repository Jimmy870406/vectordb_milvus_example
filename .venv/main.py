from milvus import Milvus
from pymilvus import FieldSchema, DataType
from embedding.openai_embedding import Openai_Embedding_Model
from embedding.sentence_transformer import Sentence_Transformer

if __name__ == "__main__":

    collection_name = "hello_milvus"
    sentence_transformers_dim = 384

    # Define document
    docs = [
        "A group of vibrant parrots chatter loudly, sharing stories of their tropical adventures.",
        "The mathematician found solace in numbers, deciphering the hidden patterns of the universe.",
        "The robot, with its intricate circuitry and precise movements, assembles the devices swiftly.",
        "The chef, with a sprinkle of spices and a dash of love, creates culinary masterpieces.",
        "The ancient tree, with its gnarled branches and deep roots, whispers secrets of the past.",
        "The detective, with keen observation and logical reasoning, unravels the intricate web of clues.",
        "The sunset paints the sky with shades of orange, pink, and purple, reflecting on the calm sea.",
        "In the dense forest, the howl of a lone wolf echoes, blending with the symphony of the night.",
        "The dancer, with graceful moves and expressive gestures, tells a story without uttering a word.",
        "In the quantum realm, particles flicker in and out of existence, dancing to the tunes of probability."
    ]

    # Define fields for our collection
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim = sentence_transformers_dim)
    ]

    milvus_instance = Milvus()
    openai_embedding_instance = Openai_Embedding_Model()
    sentence_transformer = Sentence_Transformer()

    try:
        milvus_instance.connect()

        if milvus_instance.has_collections(collection_name):
            milvus_instance.drop_collection(collection_name)

        collection = milvus_instance.create_collection(collection_name, fields, "Collection for demo purposes")
        milvus_instance.list_collections()

        ## Insert data into Milvus
        embeddings = [milvus_instance.embed(doc, sentence_transformer) for doc in docs]
        entities = [
            [str(i) for i in range(len(docs))],
            [str(doc) for doc in docs],
            embeddings
        ]

        insert_result = milvus_instance.insert(collection_name, entities)
        print(f"Insert result: {insert_result}")

        ## Create the index for Milvus, devide to 128 list
        ## nlist ->　How many cluster you want to devide
        ## nprobe -> How many cluster you want to base on to query data
        milvus_instance.create_index(collection_name, "embeddings", "IVF_FLAT", "L2", {"nlist": 128})

        ## Query data from Milvus
        query = "Give me some content about the ocean"
        query_vector = milvus_instance.embed(query, sentence_transformer)
        print(milvus_instance.search(collection_name, [query_vector], "embeddings", {"metric_type": "L2", "params": {"nprobe": 10}}))

        milvus_instance.drop_all(collection_name, f'pk in ["{insert_result.primary_keys[0]}", "{insert_result.primary_keys[1]}"]')

    except Exception as e:
        print(f"An error occurred: {e}")