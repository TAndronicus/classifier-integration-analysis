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

    def test_should_sort_files(self):
        # given
        filenames = ['biodeg.scsv', 'bupa.dat', 'cryotherapy.xlsx', 'data_banknote_authentication.csv',
                     'haberman.dat', 'ionosphere.dat', 'meter_a.tsv', 'pop_failures.tsv', 'seismic_bumps.dat',
                     'twonorm.dat', 'wdbc.dat', 'wisconsin.dat']
        expected_filenames = FileHelper.FILENAMES
        for expected_filename in expected_filenames:
            if expected_filename not in filenames:
                expected_filenames.remove(expected_filename)
        # when
        filenames_sorted = FileHelper.sort_filenames_by_size(filenames)
        # then
        for expected_filename, filename_sorted in zip(expected_filenames, filenames_sorted):
            self.assertEqual(expected_filename, filename_sorted)


if __name__ == '__main__':
    unittest.main()
