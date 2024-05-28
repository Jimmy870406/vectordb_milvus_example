from abc import ABC, abstractmethod

class Embedding_Model(ABC):
    @abstractmethod
    def embedding(self):
        pass