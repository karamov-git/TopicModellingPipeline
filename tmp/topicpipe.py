import numpy as np
import pandas as pd

from text_vectorizer import AbstractTextVectorizer
from models import AbstractTopicModellingModel
from text_preprocessors import AbstractTextPreprocessors


class AbstractTopicPipeline:

    def __init__(self, input_preprocessing: AbstractTextPreprocessors, vectorization: AbstractTextVectorizer,
                 model: AbstractTopicModellingModel):
        self.input_preprocessing = input_preprocessing
        self.vectorization = vectorization
        self.model = model

    def run(self, document_collections: pd.Series) -> np.ndarray:
        pass


class TopicPipeline(AbstractTopicPipeline):
    def __init__(self, input_preprocessing: AbstractTextPreprocessors, vectorization: AbstractTextVectorizer,
                 model: AbstractTopicModellingModel):
        super().__init__(input_preprocessing, vectorization, model)

    def run(self, document_collections: pd.Series) -> np.ndarray:
        tidy_data = self.input_preprocessing(document_collections)
        bag_of_words = self.vectorization(tidy_data)
        self.model.fit()
        return self.model.transform(bag_of_words)
