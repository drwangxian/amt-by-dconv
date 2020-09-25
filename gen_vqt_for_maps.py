"""
This script generates VQT spectrograms for the MAPS dataset.

How to use this code
1. Create an environment variable named maps_vqt pointing to the folder where the generated spectrograms will be stored.
   This folder will be created automatically if it does not exist beforehand.
2. Configure the following parameter:
   DEBUG: in {True, False}. If True, will run in a debug mode where only very few recordings will be run. The debug
          mode enables you to quickly check if the script can run correctly.
"""

DEBUG = False

import re
import os
import numpy as np
import glob
import collections
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import matlab.engine
import soundfile

FOLDER = os.environ['maps_vqt']


class MiscFns(object):
    """Miscellaneous functions"""
    @staticmethod
    def filename_to_id(filename):
        """Translate a .wav or .mid path to a MAPS sequence id."""
        return re.match(r'.*MUS-(.+)_[^_]+\.\w{3}',
                        os.path.basename(filename)).group(1)

    @staticmethod
    def split_train_valid_and_test_files_fn():
        test_dirs = ['ENSTDkCl_2/MUS', 'ENSTDkAm_2/MUS']
        train_dirs = ['AkPnBcht_2/MUS', 'AkPnBsdf_2/MUS', 'AkPnCGdD_2/MUS', 'AkPnStgb_2/MUS',
                      'SptkBGAm_2/MUS', 'SptkBGCl_2/MUS', 'StbgTGd2_2/MUS']
        maps_dir = os.environ['maps']

        test_files = []
        for directory in test_dirs:
            path = os.path.join(maps_dir, directory)
            path = os.path.join(path, '*.wav')
            wav_files = glob.glob(path)
            test_files += wav_files

        test_ids = set([MiscFns.filename_to_id(wav_file) for wav_file in test_files])
        assert len(test_ids) == 53

        training_files = []
        validation_files = []
        for directory in train_dirs:
            path = os.path.join(maps_dir, directory)
            path = os.path.join(path, '*.wav')
            wav_files = glob.glob(path)
            for wav_file in wav_files:
                me_id = MiscFns.filename_to_id(wav_file)
                if me_id not in test_ids:
                    training_files.append(wav_file)
                else:
                    validation_files.append(wav_file)

        assert len(training_files) == 139 and len(test_files) == 60 and len(validation_files) == 71

        return dict(training=training_files, test=test_files, validation=validation_files)

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
        self.file_names = MiscFns.split_train_valid_and_test_files_fn()
        if self.debug_mode:
            self.file_names['training'] = self.file_names['training'][:3]
            self.file_names['validation'] = self.file_names['validation'][:2]
            self.file_names['test'] = self.file_names['test'][:2]


class GenVQT(object):
    def __init__(self, config):
        self.config = config
        folder = self.config.folder
        if not os.path.isdir(folder):
            logging.info('folder {} does not exist, so create one'.format(folder))
            os.system('mkdir {}'.format(folder))
        self.folder = folder

    def gen_vqt_fn(self):
        for training_valid_or_test in ('training', 'test', 'validation'):
            file_names = self.config.file_names[training_valid_or_test]
            logging.debug('{} - generate vqts'.format(training_valid_or_test))
            _num_recs = len(file_names)
            for file_idx, wav_file_name in enumerate(file_names):
                base_name = os.path.basename(wav_file_name)[:-4]
                logging.info('{}/{} - {}'.format(file_idx + 1, _num_recs, base_name))

                vqt_output_file_name = base_name + '.vqt'
                vqt_output_file_name = os.path.join(self.folder, vqt_output_file_name)
                if os.path.isfile(vqt_output_file_name):
                    try:
                        name_returned, _ = MiscFns.load_np_array_from_file_fn(vqt_output_file_name)
                        if name_returned == base_name:
                            logging.info('{} already exists so skip this recording'.format(vqt_output_file_name))
                            continue
                        else:
                            logging.info('{} already exists but seems cracked so re-generate it'.format(
                                vqt_output_file_name))
                    except Exception as _e:
                        print _e
                        logging.info(
                            '{} already exists but seems cracked so re-generate it'.format(vqt_output_file_name))

                if file_idx == 0:
                    sg = self._vqt_fn(wav_file_name, err_db=True)
                else:
                    sg = self._vqt_fn(wav_file_name)

                assert sg.shape[1:] == (336,)

                MiscFns.save_np_array_to_file_fn(
                    file_name=vqt_output_file_name, output=sg, rec_name=base_name)
                if file_idx == 0:
                    name_returned, sg_returned = MiscFns.load_np_array_from_file_fn(vqt_output_file_name)
                    assert name_returned == base_name
                    assert np.array_equal(sg, sg_returned)

    def _vqt_fn(self, wav_file, err_db=False):
        with matlab.engine.start_matlab(option='-nojvm -nodesktop') as mat_eng:
            if not err_db:
                coeffs = mat_eng.vqt_fn('wav_file', wav_file)
            else:
                coeffs, err_db = mat_eng.vqt_fn('wav_file', wav_file, nargout=2)
                assert err_db >= 290.
                logging.info('vqt accuracy - {} dB'.format(err_db))
            coeffs = np.array(coeffs._data, dtype=np.float32).reshape(coeffs.size, order='F')
        num_frames = (soundfile.info(wav_file).frames + 63) // 64
        num_frames = (num_frames + 21) // 22
        assert coeffs.shape == (num_frames, 336)
        coeffs = np.require(coeffs, dtype=np.float32, requirements=['C', 'O'])

        return coeffs


def main():
    gen_vqt_ins = GenVQT(config=Config())
    gen_vqt_ins.gen_vqt_fn()


if __name__ == '__main__':
    main()









