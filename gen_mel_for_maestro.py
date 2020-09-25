"""
This script generates mel spectrograms for the MAESTRO dataset.

How to use this code
1. Create an environment variable named maestro_mel pointing to the folder
   where the generated spectrograms will be stored. This folder will be created automatically if it does not exist.
2. Configure the following parameter:
    DEBUG: in {True, False}. If True, will run in a debug mode where only very few recordings will be run. The debug
               mode enables you to quickly check if the script can run correctly.
"""
import os

DEBUG = False
FOLDER = os.environ['maestro_mel']

import re
import numpy as np
import glob
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import soundfile
import csv
import madmom
import librosa


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

        year_name_split_list = MiscFns.get_maestro_year_name_split_list_fn()

        years = [it[0] for it in year_name_split_list]
        years = set(years)

        for year in years:
            year_dir = os.path.join(self.folder, year)
            if not os.path.isdir(year_dir):
                logging.info('folder {} does not exist, create it'.format(year_dir))
                os.system('mkdir -p {}'.format(year_dir))
            else:
                logging.info('folder {} already exists'.format(year_dir))

        self.year_name_list = [[it[0], it[1]] for it in year_name_split_list]

        logging.info('folder for mel sgs - {}'.format(self.folder))
        logging.info('num of recs - {}'.format(len(self.year_name_list)))

        if self.debug_mode:
            tmp = np.random.choice(len(self.year_name_list), 3, replace=False)
            self.year_name_list = [self.year_name_list[ii] for ii in tmp]

        self.log_filter_bank = MiscFns.log_filter_bank_fn()


class GenMelSg(object):
    def __init__(self):
        self.config = Config()

    def gen_mel_sg_fn(self):
        num_recs = len(self.config.year_name_list)
        mel_folder = self.config.folder
        for rec_idx, (year, rec_name) in enumerate(self.config.year_name_list):
            logging.info('{}/{} - {}'.format(rec_idx + 1, num_recs, rec_name))
            mel_file = os.path.join(mel_folder, year, rec_name + '.mel')

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

            mel = self._gen_mel_sg_fn(year, rec_name)
            MiscFns.save_np_array_to_file_fn(mel_file, mel, rec_name)
            if rec_idx == 0:
                _rec_name, _mel = MiscFns.load_np_array_from_file_fn(mel_file)
                assert _rec_name == rec_name
                assert np.array_equal(mel, _mel)
        logging.info('done')

    def _gen_mel_sg_fn(self, year, rec_name):
        wav_file = os.path.join(os.environ['maestro'], year, rec_name + '.wav')
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









