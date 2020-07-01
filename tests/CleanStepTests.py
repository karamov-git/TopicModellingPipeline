import unittest
import text.clean_step as text_clean
import pandas as pd


class BaseCleanStepTests(unittest.TestCase):

    def test_should_raise_value_error_when_call_contain_more_one_argument(self):
        baseCleanStep = text_clean.BaseCleanStep()
        arguments = ['first', 'second']
        with self.assertRaises(ValueError):
            baseCleanStep(*arguments)

    def test_should_raise_value_error_when_call_contain_less_one_argument(self):
        baseCleanStep = text_clean.BaseCleanStep()
        arguments = []
        with self.assertRaises(ValueError):
            baseCleanStep(*arguments)

    def test_should_raise_value_error_when_argument_not_instance_if_pdSeries(self):
        baseCleanStep = text_clean.BaseCleanStep()
        arguments = ['not pd.Series']
        with self.assertRaises(ValueError):
            baseCleanStep(*arguments)

    def test_should_return_argument_when_it_correct(self):
        baseCleanStep = text_clean.BaseCleanStep()
        documentCollections = pd.Series()
        arguments = [documentCollections]
        actualValue = baseCleanStep(*arguments)
        self.assertTrue(actualValue.equals(documentCollections))


class HTMLCleanStepTests(unittest.TestCase):

    def test_should_extract_raw_text_from_html(self):
        html_clean_step = text_clean.HTMLCleanStep()
        html = """<div><p>{0}</p>\n<p>{1}</p></div>""".format('needed', 'data')
        expected_data = pd.Series(['needed data'])
        actual_value = html_clean_step(pd.Series([html]))
        self.assertTrue(actual_value.equals(expected_data))


class NtlkTokenizeCleanStep(unittest.TestCase):

    def test_should_tokenize_text(self):
        data = ['first document', 'second document']
        tokenize_clean_step = text_clean.NtlkTokenizeCleanStep()
        expected_data = pd.Series([data[0].split(), data[1].split()])
        actual_data = tokenize_clean_step(pd.Series(data))
        self.assertTrue(expected_data.equals(actual_data))


class ToLowerCaseCleanStep(unittest.TestCase):

    def test_should_to_lower_case_documents(self):
        data = ['First DOCUMENT', 'SeConD documenT']
        expected_data = pd.Series([data[0].lower(), data[1].lower()])
        to_lower_clean_step = text_clean.ToLowerCaseDocumentCleanStep()
        actual_data = to_lower_clean_step(pd.Series(data))
        self.assertTrue(actual_data.equals(expected_data))


class StopWordsCleanStepTest(unittest.TestCase):

    def test_should_raise_exception_when_series_element_not_list(self):
        data = ['not split document']
        stop_words_clean_step = text_clean.StopWordsCleanStep(['some', 'words'])
        with self.assertRaises(ValueError):
            stop_words_clean_step(pd.Series(data))

    def test_should_remove_stop_words_from_documents(self):
        data = [['first', 'stop', 'document'], ['second', 'document', 'word']]
        expected_data = pd.Series([['first', 'document'], ['second', 'document']])
        stop_words_clean_step = text_clean.StopWordsCleanStep(['stop', 'word'])
        actual_data = stop_words_clean_step(pd.Series(data))
        self.assertTrue(actual_data.equals(expected_data))


class FilterEmptyDocumentsCleanStepTests(unittest.TestCase):

    def test_should_remove_empty_documents(self):
        data = ['first document', '', 'second document', None]
        expected_data = pd.Series(['first document', 'second document'])
        step = text_clean.FilterEmptyDocumentsCleanStep()
        actual_data = step(pd.Series(data)).reset_index(drop=True)
        self.assertTrue(actual_data.equals(expected_data))

    def test_after_removing_not_drop_index(self):
        data = ['first document', '', 'second document']
        expected_index = [0, 2]
        step = text_clean.FilterEmptyDocumentsCleanStep()
        actual_index = step(pd.Series(data)).index.to_list()
        self.assertTrue(expected_index == actual_index)


class ApplyFunctionForDocumentCleanStepTests(unittest.TestCase):

    def test_should_apply_some_function_above_each_document(self):
        def some_function(document):
            return document.upper()

        data = ['first document', 'second document']
        expected_data = pd.Series([some_function(data[0]), some_function(data[1])])
        step = text_clean.ApplyFunctionForDocumentCleanStep(some_function)
        actual_data = step(pd.Series(data))
        self.assertTrue(actual_data.equals(expected_data))


class RuLemmatizationCleanStepTests(unittest.TestCase):

    def test_should_normalize_words(self):
        data = ['первые документам', 'второй документов']
        expected_data = pd.Series(['первый документ', 'второй документ'])
        step = text_clean.RuLemmatizationCleanStep()
        actual_data = step(pd.Series(data))
        self.assertTrue(actual_data.equals(expected_data))

    def test_should_ignore_not_russian_words(self):
        data = ['I saw их днём']
        expected_data = pd.Series(['I saw их день'])
        step = text_clean.RuLemmatizationCleanStep()
        actual_data = step(pd.Series(data))
        self.assertTrue(actual_data.equals(expected_data))


if __name__ == '__main__':
    unittest.main()
