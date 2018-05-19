import unittest
import FileHelper


class FileHelperTest(unittest.TestCase):

    def test_should_match_multiple_files(self):
        # given
        filename_raw = 'b'
        # when
        # then
        with self.assertRaisesRegex(FileNotFoundError, 'ambiguous'):
            FileHelper.get_full_filename(filename_raw)

    def test_should_not_match_any_filename(self):
        # given
        filename_raw = 'a'
        # when
        # then
        with self.assertRaisesRegex(FileNotFoundError, 'not found'):
            FileHelper.get_full_filename(filename_raw)

    def test_should_match_file(self):
        # given
        filename_raw = 'c'
        expected_filename = 'cryotherapy.xlsx'
        # when
        filename = FileHelper.get_full_filename(filename_raw)
        # then
        self.assertEqual(expected_filename, filename)

    def test_should_find_ambiguous_filename(self):
        # given
        filenames_raw = ['c', 'b']
        # then
        with self.assertRaisesRegex(FileNotFoundError, 'ambiguous.*b'):
            FileHelper.prepare_filenames(filenames_raw)

    def test_not_should_find_filenames(self):
        # given
        filenames_raw = ['c', 'a']
        # then
        with self.assertRaisesRegex(FileNotFoundError, 'not found.*a'):
            FileHelper.prepare_filenames(filenames_raw)

    def test_should_find_filenames(self):
        # given
        filenames_raw = ['d', 'c']
        expected_filenames = ['data_banknote_authentication.csv', 'cryotherapy.xlsx']
        # when
        filenames = FileHelper.prepare_filenames(filenames_raw)
        # then
        for expected_filename, filename in zip(expected_filenames, filenames):
            self.assertEqual(expected_filename, filename)


if __name__ == '__main__':
    unittest.main()
