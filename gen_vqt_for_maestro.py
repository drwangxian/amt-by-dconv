"""
This script generates VQT spectrograms for the MAESTRO dataset.

How to use this code
1. Create an environment variable named maestro_vqt pointing to the folder where the generated spectrograms will be
   stored. This folder will be created automatically if it does not exist beforehand.
2. Configure the following parameter:
   DEBUG: in {True, False}. If True, will run in a debug mode where only very few recordings will be run. The debug
          mode enables you to quickly check if the script can run correctly.
"""

DEBUG = False

import os
FOLDER = os.environ['maestro_vqt']
import re
import numpy as np
import glob
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import matlab.engine
import soundfile
import csv


class MiscFns(object):
    """Miscellaneous functions"""
    @staticmethod
    def get_maestro_year_name_split_list_fn():
        csv_file = glob.glob(os.path.join(os.environ['maestro'], '*.csv'))
        assert len(csv_file) == 1
        csv_file = csv_file[0]

        name_to_idx_dict = dict(
            canonical_composer=0,
            canonical_title=1,
            split=2,
            year=3,
            midi_filename=4,
            audio_filename=5,
            duration=6
        )

        year_name_split_list = []
        with open(csv_file) as csv_fh:
            csv_reader = csv.reader(csv_fh)
            head_row = next(csv_reader)
            for field_name in head_row:
                assert field_name in name_to_idx_dict
            get_year = re.compile(r'^(2[0-9]{3})/.*')
            for row in csv_reader:
                mid_file = row[name_to_idx_dict['midi_filename']]
                audio_file = row[name_to_idx_dict['audio_filename']]
                year = row[name_to_idx_dict['year']]
                assert get_year.match(mid_file).group(1) == get_year.match(audio_file).group(1) == year
                mid_base_name = os.path.basename(mid_file)[:-5]
                audio_base_name = os.path.basename(audio_file)[:-4]
                assert mid_base_name == audio_base_name
                rec_name = mid_base_name
                assert mid_file == os.path.join(year, rec_name + '.midi')
                assert audio_file == os.path.join(year, rec_name + '.wav')
                year_name_split_list.append([year, rec_name, row[name_to_idx_dict['split']]])
            assert len(year_name_split_list) == 1184
        return year_name_split_list

    @staticmethod
    def save_np_array_to_file_fn(file_name, output, rec_name):
        with open(file_name, 'wb') as fh:
            fh.write(b'{:s}'.format(rec_name))
            fh.write(b' ')
            fh.write(b'{:s}'.format(output.dtype))
            for dim_size in output.shape:
                fh.write(' ')
                fh.write('{:d}'.format(dim_size))
            fh.write('\n')
            fh.write(output.data)
            fh.flush()
            os.fsync(fh.fileno())

    @staticmethod
    def load_np_array_from_file_fn(file_name):
        with open(file_name, 'rb') as fh:
            first_line = str(fh.readline()).split()
            rec_name = first_line[0]
            dtype = first_line[1]
            dim = first_line[2:]
            dim = [int(_item) for _item in dim]
            output = np.frombuffer(fh.read(), dtype=dtype).reshape(*dim)
            return rec_name, output


class Config(object):
    def __init__(self):
        self.debug_mode = DEBUG
        self.folder = FOLDER

        year_name_split_list = MiscFns.get_maestro_year_name_split_list_fn()

        years = [it[0] for it in year_name_split_list]
        years = set(years)

        for year in years:
            year_dir = os.path.join(self.folder, year)
            if not os.path.isdir(year_dir):
                logging.info('folder {} does not exist, crete it'.format(year_dir))
                os.system('mkdir -p {}'.format(year_dir))
            else:
                logging.info('folder {} already exists'.format(year_dir))

        self.year_name_list = [it[:2] for it in year_name_split_list]

        logging.info('folder for vqt - {}'.format(self.folder))
        logging.info('num of recs - {}'.format(len(self.year_name_list)))

        if self.debug_mode:
            tmp = [1, 100, 200]
            self.year_name_list = [self.year_name_list[ii] for ii in tmp]


class GenVQT(object):
    def __init__(self):
        self.config = Config()

    def gen_vqt_fn(self):
        num_recs = len(self.config.year_name_list)
        wav_folder = os.environ['maestro']
        vqt_folder = self.config.folder
        for rec_idx, (year, rec_name) in enumerate(self.config.year_name_list):
            logging.info('{}/{} - {}'.format(rec_idx + 1, num_recs, rec_name))
            wav_file = os.path.join(wav_folder, year, rec_name + '.wav')
            vqt_file = os.path.join(vqt_folder, year, rec_name + '.vqt')

            if os.path.isfile(vqt_file):
                try:
                    _rec_name, _ = MiscFns.load_np_array_from_file_fn(vqt_file)
                    if _rec_name == rec_name:
                        logging.info('{} already exists so skip this recording'.format(vqt_file))
                        continue
                    else:
                        logging.info(
                            '{} already exists but seems cracked so re-generate it'.format(vqt_file))
                except Exception as _e:
                    logging.info(_e)
                    logging.info('{} already exists but seems cracked so re-generate it'.format(vqt_file))

            vqt = self._vqt_fn(wav_file, err_db=True if rec_idx == 0 else False)
            MiscFns.save_np_array_to_file_fn(vqt_file, vqt, rec_name)
            if rec_idx == 0:
                _rec_name, _vqt = MiscFns.load_np_array_from_file_fn(vqt_file)
                assert _rec_name == rec_name
                assert np.array_equal(vqt, _vqt)
        logging.info('done')

    def _vqt_fn(self, wav_file, err_db=False):
        with matlab.engine.start_matlab(option='-nojvm -nodesktop') as mat_eng:
            if not err_db:
                coeffs = mat_eng.vqt_fn('wav_file', wav_file)
            else:
                coeffs, err_db = mat_eng.vqt_fn('wav_file', wav_file, nargout=2)
                assert err_db >= 285.
                logging.info('vqt accuracy - {} dB'.format(err_db))
            coeffs = np.array(coeffs._data, dtype=np.float32).reshape(coeffs.size, order='F')

        wav_info = soundfile.info(wav_file)
        sr = 44100
        assert wav_info.samplerate >= sr
        if wav_info.samplerate > sr:
            num_frames = (wav_info.frames * sr + wav_info.samplerate - 1) // wav_info.samplerate
        else:
            num_frames = wav_info.frames
        num_frames = (num_frames + 63) // 64
        num_frames = (num_frames + 21) // 22

        assert coeffs.shape == (num_frames, 336)
        coeffs = np.require(coeffs, dtype=np.float32, requirements=['C', 'O'])

        return coeffs


def main():
    GenVQT().gen_vqt_fn()


if __name__ == '__main__':
    main()









