import re

import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem


class BaseCleanStep:

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        if len(args) != 1:
            raise ValueError('arguments should contain single parameter')
        documents_collection = args[0]
        if not isinstance(documents_collection, pd.Series):
            raise ValueError('argument should instance of pandas.Series')

        return documents_collection


class HTMLCleanStep(BaseCleanStep):

    def __init__(self):
        super().__init__()
        pass

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(*args)
        return documents_collection.apply(self.__extract_row_text_from_html)

    @staticmethod
    def __extract_row_text_from_html(text):
        soup = BeautifulSoup(text, 'html.parser')
        html_free = soup.get_text(strip=True, separator=' ')
        return html_free


class NtlkTokenizeCleanStep(BaseCleanStep):

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(*args)
        return documents_collection.apply(word_tokenize)


class ToLowerCaseDocumentCleanStep(BaseCleanStep):

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(*args)
        return documents_collection.apply(lambda document: document.lower())


class RegexCleanStep(BaseCleanStep):
    def __init__(self, pattern, flags=0):
        super().__init__()
        self.regex = re.compile(pattern, flags)

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(*args)
        return documents_collection.apply(lambda document: self.regex.sub(' ', document))


class StopWordsCleanStep(BaseCleanStep):

    def __init__(self, stop_words: list):
        super().__init__()
        self.stop_words = stop_words

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(*args)
        if not isinstance(documents_collection[0], list):
            raise ValueError('This step should take series of list words from documents, but get {}'.format(
                type(documents_collection[0])))
        return documents_collection.apply(
            lambda document: list(filter(lambda word: word not in self.stop_words, document)))


class FilterEmptyDocumentsCleanStep(BaseCleanStep):

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(*args)
        return documents_collection[(documents_collection.notnull()) & (documents_collection != '')]


class ApplyFunctionForDocumentCleanStep(BaseCleanStep):

    def __init__(self, f):
        super().__init__()
        self.f = f

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(*args)
        return documents_collection.apply(self.f)


class RuLemmatizationCleanStep(BaseCleanStep):

    def __init__(self, faster=False):
        super().__init__()
        self.analyzer = Mystem()
        self.faster = faster

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(*args)
        if self.faster:
            return self.__lemmatize_faster(documents_collection)
        return documents_collection.apply(self.__lemmatize)

    def __lemmatize_faster(self, documents: pd.Series):

        def lemmatize_merged_documents_by_br_and_split_after(texts, analyzer):
            lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
            txtpart = lol(texts, 1000)
            res = []
            for txtp in txtpart:
                alltexts = ' '.join([txt + ' br ' for txt in txtp])

                words = analyzer.lemmatize(alltexts)

                doc = []
                for txt in words:
                    if txt != '\n' and txt.strip() != '':
                        if txt == 'br':
                            res.append(doc)
                            doc = []
                        else:
                            doc.append(txt)
            return res

        cleaned_data = lemmatize_merged_documents_by_br_and_split_after(documents.to_list(), self.analyzer)
        return pd.Series(cleaned_data, index=documents.index)

    def __lemmatize(self, document):
        lemmatization_document = [word for word in self.analyzer.lemmatize(document) if
                                  word not in ['', '\n', '\t', None]]
        return ''.join(lemmatization_document)
