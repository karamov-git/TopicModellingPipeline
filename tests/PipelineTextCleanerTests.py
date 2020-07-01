import unittest
import text.clean_step as text_clean
from text_preprocessors import PipelineTextCleaner
import pandas as pd


class PipelineTextCleanerTests(unittest.TestCase):
    def test_should_clean_use_pipeline(self):
        data = [
            '<div><p>1.Это первый документы</p>,& не *содержат* лишние знаки и <a href=\'some_link\'>html?!!!</div>',
            '\u20A9Это второй \u2022документ, 	\u2117удалите не нужные \u221Eюникод символы',
            'вСе дОкумЕнты должны быть в нижнем РЕГИСТРЕ',
            'И иметь нормальный форма']
        expected_data = pd.Series(['это первый документ содержать лишний знак html',
                                   'это второй документ удалять нужный юникод символ',
                                   'весь документ должный быть нижний регистр',
                                   'иметь нормальный форма'])
        pipeline = self.__get_test_pipeline()
        cleaner = PipelineTextCleaner(pipeline)
        actual_result = cleaner.clean_up(pd.Series(data))
        self.assertTrue(actual_result.equals(expected_data))

    def __get_test_pipeline(self):
        def join_tokens(tokenize_document):
            return ' '.join([word.strip() for word in tokenize_document if word not in ['', ' ']])

        return [text_clean.HTMLCleanStep(),
                text_clean.ToLowerCaseDocumentCleanStep(),
                text_clean.RegexCleanStep(r'\W|\d'),
                text_clean.RuLemmatizationCleanStep(),
                text_clean.NtlkTokenizeCleanStep(),
                text_clean.StopWordsCleanStep(['не', 'и', 'в']),
                text_clean.ApplyFunctionForDocumentCleanStep(join_tokens)]


if __name__ == '__main__':
    unittest.main()
