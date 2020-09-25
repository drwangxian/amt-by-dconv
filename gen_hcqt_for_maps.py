"""
This script generates HCQTs for the MAPS dataset.

How to use this code
1. Create an environment variable named maps_hcqt pointing to the folder where the generated spectrograms will be stored.
   This folder will be created automatically if it does not exist beforehand.
2. Configure the following parameter:
   DEBUG: in {True, False}. If True, will run in a debug mode where only very few recordings will be run. The debug
          mode enables you to quickly check if the script can run correctly.
"""
import os

DEBUG = False
FOLDER = os.environ['maps_hcqt']

import re
import numpy as np
import glob
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import librosa


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

    @staticmethod
    def hcqt_fn(wav_file):
        bins_per_octave = 60
        sr = 44100
        hop_length = 512

        y, _ = librosa.load(wav_file, sr=sr)

        FMIN = librosa.midi_to_hz([21])[0] * 2. ** (-2. / bins_per_octave)
        FMAX = librosa.midi_to_hz([136])[0] * 2. ** (2. / bins_per_octave)
        assert FMAX < sr / 2.

        cqt = librosa.cqt(
            y, sr=sr, hop_length=hop_length, fmin=FMIN,
            n_bins=(136 - 21 + 1) * 5,
            bins_per_octave=bins_per_octave
        )
        num_frames = (len(y) + hop_length - 1) // hop_length
        _num_frames = cqt.shape[1]
        assert _num_frames == num_frames or _num_frames == num_frames + 1
        num_frames = _num_frames
        assert cqt.shape == (580, num_frames) and cqt.dtype == np.complex128
        cqt = cqt.T
        cqt = 1. / 80. * librosa.amplitude_to_db(np.abs(cqt), ref=np.max) + 1.0
        h1 = cqt[:, : 440]
        h2 = cqt[:, 60: 500]
        h4 = cqt[:, 120: 560]
        hp5 = cqt[:, :380]
        hp5 = np.pad(hp5, ((0, 0), (60, 0)), mode='constant')
        assert hp5.shape == (num_frames, 440)

        h3 = librosa.cqt(
            y, sr=sr, hop_length=hop_length, fmin=FMIN * 3.,
            n_bins=440,
            bins_per_octave=60
        ).T
        assert h3.shape == (num_frames, 440)
        h3 = 1. / 80. * librosa.amplitude_to_db(np.abs(h3), ref=np.max) + 1.
        h5 = librosa.cqt(
            y, sr=sr, hop_length=hop_length, fmin=FMIN * 5.,
            n_bins=440,
            bins_per_octave=60
        ).T
        assert h5.shape == (num_frames, 440)
        h5 = 1. / 80. * librosa.amplitude_to_db(np.abs(h5), ref=np.max) + 1.

        hs = [hp5, h1, h2, h3, h4, h5]
        hs = np.stack(hs, axis=-1)
        assert hs.shape == (num_frames, 440, 6)
        hs = np.require(hs, dtype=np.float32, requirements=['O', 'C'])
        hs.flags['WRITEABLE'] = False

        return hs


class Config(object):
    def __init__(self):
        self.debug_mode = DEBUG
        self.folder = FOLDER

        if not os.path.isdir(self.folder):
            os.system('mkdir -p {}'.format(self.folder))
            logging.info('target dir {} does not exist so created one'.format(self.folder))
        else:
            logging.info('target dir {} already exists'.format(self.folder))

        split_dict = MiscFns.split_train_valid_and_test_files_fn()
        wav_files = []
        for value in split_dict.values():
            wav_files.extend(value)
        self.wav_files = wav_files
        assert len(self.wav_files) == 270

        if self.debug_mode:
            np.random.seed(123)
            tmp = np.random.choice(len(self.wav_files), 3, replace=False)
            self.wav_files = [self.wav_files[ii] for ii in tmp]

        logging.info('folder for hcqt sgs - {}'.format(self.folder))
        logging.info('num of recs - {}'.format(len(self.wav_files)))


class GenMelSg(object):
    def __init__(self):
        self.config = Config()

    def gen_mel_sg_fn(self):
        num_recs = len(self.config.wav_files)
        hcqt_folder = self.config.folder
        for rec_idx, wav_file in enumerate(self.config.wav_files):
            rec_name = os.path.basename(wav_file)[:-4]
            logging.info('{}/{} - {}'.format(rec_idx + 1, num_recs, rec_name))
            hcqt_file = os.path.join(hcqt_folder, rec_name + '.hcqt')

            if os.path.isfile(hcqt_file):
                try:
                    _rec_name, _ = MiscFns.load_np_array_from_file_fn(hcqt_file)
                    if _rec_name == rec_name:
                        logging.info('{} already exists so skip this recording'.format(hcqt_file))
                        continue
                    else:
                        logging.info(
                            '{} already exists but seems cracked so re-generate it'.format(hcqt_file))
                except Exception as _e:
                    logging.info(_e)
                    logging.info('{} already exists but seems cracked so re-generate it'.format(hcqt_file))

            hcqt = MiscFns.hcqt_fn(wav_file)
            MiscFns.save_np_array_to_file_fn(hcqt_file, hcqt, rec_name)
            if rec_idx == 0:
                _rec_name, _hcqt = MiscFns.load_np_array_from_file_fn(hcqt_file)
                assert _rec_name == rec_name
                assert np.array_equal(hcqt, _hcqt)
        logging.info('done')


def main():
    GenMelSg().gen_mel_sg_fn()


if __name__ == '__main__':
    main()









