import re

import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import pymorphy2
from pymorphy2.tagset import OpencorporaTag


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
        documents_collection = super().__call__(args)
        return documents_collection.apply(lambda document: document.lower())


class RegexCleanStep(BaseCleanStep):
    def __init__(self, pattern, flags=0):
        super().__init__()
        self.regex = re.compile(pattern, flags)

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(args)
        return documents_collection.apply(lambda document: self.regex.sub(" ", document))


class StopWordsCleanStep(BaseCleanStep):

    def __init__(self, stop_words: list):
        super().__init__()
        self.stop_words = stop_words

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(args)
        if not isinstance(documents_collection[0], list):
            raise ValueError('This step should take series of list words from documents, but get {}'.format(
                type(documents_collection[0])))
        return documents_collection.apply(
            lambda document: list(filter(lambda word: word not in self.stop_words, document)))


class FilterEmptyDocumentsCleanStep(BaseCleanStep):

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(args)
        return documents_collection[(documents_collection is not None) | (documents_collection != '')]


class ApplyFunctionForDocumentCleanStep(BaseCleanStep):

    def __init__(self, f):
        super().__init__()
        self.f = f

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(args)
        return documents_collection.apply(self.f)


class RuLemmatizationCleanStep(BaseCleanStep):

    def __init__(self, ignore_part_of_speech, ignore_no_russian_words=False):
        super().__init__()
        self.ignore_part_of_speech = ignore_part_of_speech
        self.ignore_no_russian_words = ignore_no_russian_words
        self.analyzer = pymorphy2.MorphAnalyzer()

    def __call__(self, *args, **kwargs):
        documents_collection = super().__call__(args)
        if not isinstance(documents_collection[0], list):
            raise ValueError('This step should take series of list words from documents, but get {}'.format(
                type(documents_collection[0])))
        return documents_collection.apply(self.__lemmatize)

    def __lemmatize(self, words):
        lemmatization_words = []
        for word in words:
            morph = max(self.analyzer.parse(word), key=lambda x: x.score)
            if morph.tag == OpencorporaTag('LATN') and self.ignore_no_russian_words:
                continue
            if morph.tag == OpencorporaTag('LATN'):
                lemmatization_words.append(word)
                continue
            if morph.tag.POS in self.ignore_part_of_speech:
                continue
            lemmatization_words.append(morph.normal_form)
        return lemmatization_words