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
        actual_data = tokenize_clean_step(pd.Series(data))
        expected_data = pd.Series([data[0].split(), data[1].split()])
        self.assertTrue(expected_data.equals(actual_data))


if __name__ == '__main__':
    unittest.main()
