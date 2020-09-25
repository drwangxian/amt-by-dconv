"""
This script generates STFTs for the MAPS dataset.

How to use this code
1. Create an environment variable named maps_stft pointing to the folder where the generated spectrograms will be stored.
   This folder will be created automatically if it does not exist beforehand.
2. Configure the following parameter:
   DEBUG: in {True, False}. If True, will run in a debug mode where only very few recordings will be run. The debug
          mode enables you to quickly check if the script can run correctly.
"""
import os

DEBUG = False
FOLDER = os.environ['maps_stft']

import re
import numpy as np
import glob
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import soundfile
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
    def stft_fn(samples):
        spec_stride = 22 * 64
        num_frames = (len(samples) + spec_stride - 1) // spec_stride
        stft = librosa.stft(y=samples, n_fft=spec_stride * 4, hop_length=spec_stride)
        assert num_frames <= stft.shape[1] <= num_frames + 1
        if stft.shape[1] == num_frames + 1:
            stft = stft[:, :num_frames]
        stft = np.abs(stft)
        stft = stft / float(2 * spec_stride)
        stft = stft.T
        stft = np.require(stft, dtype=np.float32, requirements=['C', 'O'])
        stft.flags['WRITEABLE'] = False

        return stft


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
            tmp = np.random.choice(len(self.wav_files), 3, replace=False)
            self.wav_files = [self.wav_files[ii] for ii in tmp]

        logging.info('folder for mel sgs - {}'.format(self.folder))
        logging.info('num of recs - {}'.format(len(self.wav_files)))


class GenMelSg(object):
    def __init__(self):
        self.config = Config()

    def gen_mel_sg_fn(self):
        num_recs = len(self.config.wav_files)
        mel_folder = self.config.folder
        for rec_idx, wav_file in enumerate(self.config.wav_files):
            rec_name = os.path.basename(wav_file)[:-4]
            logging.info('{}/{} - {}'.format(rec_idx + 1, num_recs, rec_name))
            stft_file = os.path.join(mel_folder, rec_name + '.stft')

            if os.path.isfile(stft_file):
                try:
                    _rec_name, _ = MiscFns.load_np_array_from_file_fn(stft_file)
                    if _rec_name == rec_name:
                        logging.info('{} already exists so skip this recording'.format(stft_file))
                        continue
                    else:
                        logging.info(
                            '{} already exists but seems cracked so re-generate it'.format(stft_file))
                except Exception as _e:
                    logging.info(_e)
                    logging.info('{} already exists but seems cracked so re-generate it'.format(stft_file))

            stft = self._gen_stft_fn(wav_file)
            MiscFns.save_np_array_to_file_fn(stft_file, stft, rec_name)
            if rec_idx == 0:
                _rec_name, _stft = MiscFns.load_np_array_from_file_fn(stft_file)
                assert _rec_name == rec_name
                assert np.array_equal(stft, _stft)
        logging.info('done')

    def _gen_stft_fn(self, wav_file):
        wav_info = soundfile.info(wav_file)
        assert wav_info.samplerate == 44100
        assert wav_info.subtype == 'PCM_16'
        assert wav_info.channels == 2
        samples, original_sr = soundfile.read(wav_file, dtype='int16')
        assert original_sr == wav_info.samplerate
        assert samples.shape == (wav_info.frames, 2) and samples.dtype == np.int16
        samples = np.sum(samples.astype(np.float32), axis=1) / (32768. * 2.)
        assert samples.ndim == 1
        sg = MiscFns.stft_fn(samples)

        assert sg.dtype == np.float32

        return sg


def main():
    GenMelSg().gen_mel_sg_fn()


if __name__ == '__main__':
    main()









