import configparser

class Config_Provider:
    _instance = None

    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config('config.ini')
        return cls._instance

    def _load_config(self, config_file):
        cfp = configparser.RawConfigParser()
        parsed_files = cfp.read(config_file)

        if not parsed_files:
            raise FileNotFoundError(f"Config file '{config_file}' not found or empty.")

        self.milvus_uri = cfp.get('example', 'uri')
        self.token = cfp.get('example', 'token')
        self.openai_api_key = cfp.get('example', 'openai_api_key')
        self.openai_chatgpt_key = cfp.get('example', 'openai_chatgpt_key')
