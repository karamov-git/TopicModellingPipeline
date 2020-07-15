import pandas as pd


class AbstractTextCleaner:

    def __init__(self):
        pass

    def clean_up(self, raw_documents_collection: pd.Series) -> pd.Series:
        pass

    def __call__(self, *args, **kwargs):
        pass


class PipelineTextCleaner(AbstractTextCleaner):

    def __init__(self, pipeline_steps: list):
        super().__init__()
        self.pipeline_steps = pipeline_steps

    def clean_up(self, raw_documents_collection: pd.Series) -> pd.Series:
        cleaned_data = raw_documents_collection
        for step in self.pipeline_steps:
            cleaned_data = step(cleaned_data)
            print('end step {}. len: {}'.format(type(step), len(cleaned_data)))

        return cleaned_data
