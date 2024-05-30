import argparse
import os
import sys
from milvus import Milvus
from pymilvus import FieldSchema, DataType
from embedding.openai_embedding import Openai_Embedding_Model
from embedding.sentence_transformer import Sentence_Transformer
from utility.txt_parser import extract_file_to_array

if __name__ == "__main__":

    # Create the parser, add the argument for the path
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default=None, type=str, help='The path to the text file')
    parser.add_argument('question', nargs='?', default=None, type=str, help='The question you want to ask')
    args = parser.parse_args()

    # Initialize the txt_path
    txt_path = args.path if args.path else input("Please provide the path to the text file: ")
    txt_path = txt_path if os.path.isabs(txt_path) else os.path.join(os.getcwd(), txt_path)

    # Initialize the txt_path
    question =  args.question if args.question else input("Please provide the question: ")

    if not (os.path.isfile(txt_path) and txt_path.lower().endswith('.txt')):
        print(f"Error: The path provided does not point to a valid text file: {txt_path}")
        sys.exit(1)

    # Extract the document from txt
    docs = extract_file_to_array(txt_path)
    collection_name = "hello_milvus"
    sentence_transformers_dim = 384

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

        ## Query data through question from Milvus
        query_vector = milvus_instance.embed(question, sentence_transformer)
        search_result = milvus_instance.search(collection_name, [query_vector], "embeddings", {"metric_type": "L2", "params": {"nprobe": 10}})

        for idx, answer in enumerate(search_result[0]):
            print(f"Top {idx}. {answer}")

        milvus_instance.drop_all(collection_name, f'pk in ["{insert_result.primary_keys[0]}", "{insert_result.primary_keys[1]}"]')

    except Exception as e:
        print(f"An error occurred: {e}")