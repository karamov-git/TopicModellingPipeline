import pandas as pd


class AbstractTextPreprocessors:

    def clean_up(self, raw_documents_collection: pd.Series) -> pd.Series:
        pass

    def __call__(self, *args, **kwargs):
        pass
