# Milvus Sample

I'd like to use a Python to connect to Milvus Cloud, insert data from a text file, and then find the closest answer using Sentence Transformer. Let's get started with using this sample!

## Step 1

The main entry is main.py, so the first step is to navigate to the .venv folder.

```
cd .\.venv\
```

## Step 2

You need to use your own Milvus connection URI and token. Follow the config.ini.example file to create a config.ini file where you can store your settings.

## Step 3

In this folder, you can simply run main.py. Here, you can specify the path to the text file and the question when running it from the command line. It will also prompt you for the path and question if you run it without providing any parameters.

1. Execute it with path and question.
```
python main.py sample.txt "Give me the content about sea" 
```

https://github.com/Jimmy870406/vectordb_milvus_example/assets/127382381/c1303f50-c173-4c07-aa9a-37fdb4e21293

2. Execute directly, then input the path and question.
```
python main.py
```

https://github.com/Jimmy870406/vectordb_milvus_example/assets/127382381/51d973a6-430b-409d-8311-77aaaf4c4fdb

## Step 4

**Wait and you will see the result!**
![image](https://github.com/Jimmy870406/vectordb_milvus_example/assets/127382381/b380a0f8-c094-4a00-b17c-3e3c80453066)


