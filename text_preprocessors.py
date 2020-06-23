import pandas as pd
from bs4 import BeautifulSoup


class AbstractTextCleaner:

    def __init__(self):
        pass

    def clean_up(self, raw_documents_collection: pd.Series) -> pd.Series:
        pass

    def __call__(self, *args, **kwargs):
        pass


class AbstractCleanStep:

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class HTMLCleanStep(AbstractCleanStep):

    def __init__(self):
        super().__init__()
        pass

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError('argument should contain single parameter')
        documents_collection = args[0]
        if not isinstance(documents_collection, pd.Series):
            raise ValueError('argument should instance of pandas.Series')

        return documents_collection.apply(self.__extract_row_text_from_html)

    @staticmethod
    def __extract_row_text_from_html(text):
        soup = BeautifulSoup(text, 'html.parser')
        html_free = soup.get_text(strip=True, separator=' ')
        return html_free


class PipelineTextCleaner(AbstractTextCleaner):

    def __init__(self, pipeline_steps: list):
        super().__init__()
        self.pipeline_steps = pipeline_steps

    def clean_up(self, raw_documents_collection: pd.Series) -> pd.Series:
        cleaned_data = raw_documents_collection
        for step in self.pipeline_steps:
            cleaned_data = step(cleaned_data)

        return cleaned_data
