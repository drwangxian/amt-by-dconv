"""
This script generates mel spectrograms for the MAPS dataset.

How to use this code
1. Create an environment variable named maps_mel pointing to the folder
   where the generated spectrograms will be stored. This folder will be created automatically if it does not exist.
2. Configure the following parameter:
    DEBUG: in {True, False}. If True, will run in a debug mode where only very few recordings will be run. The debug
               mode enables you to quickly check if the script can run correctly.
"""
import os

DEBUG = False
FOLDER = os.environ['maps_mel']

import re
import numpy as np
import glob
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import soundfile
import madmom
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
    def log_filter_bank_fn():
        log_filter_bank_basis = madmom.audio.filters.LogarithmicFilterbank(
            bin_frequencies=librosa.fft_frequencies(sr=16000, n_fft=2048),
            num_bands=48,
            fmin=librosa.midi_to_hz([27])[0],
            fmax=librosa.midi_to_hz([114])[0] * 2. ** (1. / 48)
        )
        log_filter_bank_basis = np.array(log_filter_bank_basis)
        assert log_filter_bank_basis.shape[1] == 229
        assert np.abs(np.sum(log_filter_bank_basis[:, 0]) - 1.) < 1e-3
        assert np.abs(np.sum(log_filter_bank_basis[:, -1]) - 1.) < 1e-3

        return log_filter_bank_basis

    @staticmethod
    def spectrogram_fn(samples, log_filter_bank_basis, spec_stride):
        spec_stride = int(spec_stride)
        num_frames = (len(samples) + spec_stride - 1) // spec_stride
        stft = librosa.stft(y=samples, n_fft=2048, hop_length=spec_stride)
        assert num_frames <= stft.shape[1] <= num_frames + 1
        if stft.shape[1] == num_frames + 1:
            stft = stft[:, :num_frames]
        stft = np.abs(stft)
        stft = stft / 1024.
        assert stft.dtype == np.float32
        stft = 20. * np.log10(stft + 1e-7) + 140.
        lm_mag = np.dot(stft.T, log_filter_bank_basis)
        assert lm_mag.shape[1] == 229
        lm_mag = np.require(lm_mag, dtype=np.float32, requirements=['C', 'O'])
        lm_mag.flags['WRITEABLE'] = False

        return lm_mag


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

        self.log_filter_bank = MiscFns.log_filter_bank_fn()


class GenMelSg(object):
    def __init__(self):
        self.config = Config()

    def gen_mel_sg_fn(self):
        num_recs = len(self.config.wav_files)
        mel_folder = self.config.folder
        for rec_idx, wav_file in enumerate(self.config.wav_files):
            rec_name = os.path.basename(wav_file)[:-4]
            logging.info('{}/{} - {}'.format(rec_idx + 1, num_recs, rec_name))
            mel_file = os.path.join(mel_folder, rec_name + '.mel')

            if os.path.isfile(mel_file):
                try:
                    _rec_name, _ = MiscFns.load_np_array_from_file_fn(mel_file)
                    if _rec_name == rec_name:
                        logging.info('{} already exists so skip this recording'.format(mel_file))
                        continue
                    else:
                        logging.info(
                            '{} already exists but seems cracked so re-generate it'.format(mel_file))
                except Exception as _e:
                    logging.info(_e)
                    logging.info('{} already exists but seems cracked so re-generate it'.format(mel_file))

            mel = self._gen_mel_sg_fn(wav_file)
            MiscFns.save_np_array_to_file_fn(mel_file, mel, rec_name)
            if rec_idx == 0:
                _rec_name, _mel = MiscFns.load_np_array_from_file_fn(mel_file)
                assert _rec_name == rec_name
                assert np.array_equal(mel, _mel)
        logging.info('done')

    def _gen_mel_sg_fn(self, wav_file):
        wav_info = soundfile.info(wav_file)
        assert wav_info.samplerate in (44100, 48000)
        assert wav_info.subtype == 'PCM_16'
        assert wav_info.channels == 2
        samples, original_sr = soundfile.read(wav_file, dtype='int16')
        assert original_sr == wav_info.samplerate
        assert samples.shape == (wav_info.frames, 2) and samples.dtype == np.int16
        samples = np.sum(samples.astype(np.float32), axis=1) / (32768. * 2.)
        assert samples.ndim == 1
        samples = librosa.resample(samples, wav_info.samplerate, 16000)
        _len = int(np.ceil(wav_info.frames * 16000. / wav_info.samplerate))
        assert _len == len(samples)
        sg = MiscFns.spectrogram_fn(samples, self.config.log_filter_bank, 512)

        assert sg.dtype == np.float32

        return sg


def main():
    GenMelSg().gen_mel_sg_fn()


if __name__ == '__main__':
    main()









