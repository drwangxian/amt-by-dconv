"""
This script implements the onset detector of our AMT system.

How to use this code
1. Download the datasets and generate spectrograms.
2. Create a folder, e.g., maps/onset if you want to train on the MAPS dataset, and copy this
   script to the folder. By default, the checkpoints will be saved in folder ./saved_model, and the statistics and
   other information will be saved in folder ./tb_d0. You can view the outputs with tensorboard.
3. Configure the following parameters:
    DEBUG: in {True, False}. If True, will run in a debug mode where only very few recordings will be run. The debug
           mode enables you to quickly check if the script can run correctly.
    GPU_ID: in {0, 1, ..., n - 1} where n is the number of GPUs available.
    TRAINING_DATASET: in {'maps', 'maestro'}. This sets the dataset for training.
    PREFETCH_RATIO: in (0, 1]. The data for validation and test will be all loaded into memory. Only a fraction of the
                    data for training will be loaded into memory. This sets the fraction.
4. Refer to class Config for more options, e.g., continue training from a saved checkpoint, or run in inference mode.
"""


from __future__ import print_function
import numpy as np

DEBUG = False
GPU_ID = 1
TRAINING_DATASET = 'maps'
PREFETCH_RATIO = 1

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import glob
import re

from argparse import Namespace
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import datetime
import magenta.music
import soundfile
import mido
import collections
import csv
from scipy.ndimage.filters import maximum_filter1d
import mir_eval
import librosa


# miscellaneous functions
class MiscFns(object):
    """Miscellaneous functions"""

    @staticmethod
    def filename_to_id(filename):
        """Translate a .wav or .mid path to a MAPS sequence id."""
        return re.match(r'.*MUS-(.+)_[^_]+\.\w{3}',
                        os.path.basename(filename)).group(1)

    @staticmethod
    def times_to_frames_fn(sr, spec_stride, start_time, end_time):
        sr = int(sr)
        spec_stride = int(spec_stride)
        assert (sr, spec_stride) in [(16000, 512), (44100, 22 * 64)]
        assert spec_stride % 2 == 0
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        start_frame = (start_sample + spec_stride // 2) // spec_stride
        end_frame = (end_sample + spec_stride // 2 - 1) // spec_stride

        return start_frame, end_frame + 1

    @staticmethod
    def label_fn(sr, mid_file_name, num_frames, spec_stride):
        sr = int(sr)
        spec_stride = int(spec_stride)
        assert (sr, spec_stride) in [(16000, 512), (44100, 22 * 64)]
        onset_matrix = np.zeros((num_frames, 88), dtype=np.bool_)
        note_seq = magenta.music.midi_file_to_note_sequence(mid_file_name)
        note_seq = magenta.music.apply_sustain_control_changes(note_seq)
        for note in note_seq.notes:
            assert 21 <= note.pitch <= 108
            note_start_frame, note_end_frame = MiscFns.times_to_frames_fn(
                sr=sr,
                spec_stride=spec_stride,
                start_time=note.start_time,
                end_time=note.end_time
            )
            onset_matrix[note_start_frame: min(note_end_frame, note_start_frame + 2), note.pitch - 21] = True

        return onset_matrix

    @staticmethod
    def acoustic_model_fn(spec_batch, is_training, use_feature, trainable):
        assert tf.get_variable_scope().name != ''
        spec_batch.set_shape([None, None, 336])
        assert all(isinstance(v, bool) for v in (is_training, use_feature, trainable))
        outputs = spec_batch[..., None]

        outputs = slim.conv2d(
            scope='C_0',
            inputs=outputs,
            num_outputs=32,
            kernel_size=3,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training, trainable=trainable),
            trainable=trainable
        )

        outputs = slim.conv2d(
            scope='C_1',
            inputs=outputs,
            num_outputs=32,
            kernel_size=3,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training, trainable=trainable),
            trainable=trainable
        )
        outputs = slim.dropout(scope='DO_1', inputs=outputs, keep_prob=.8, is_training=is_training)

        outputs = slim.conv2d(
            scope='C_2',
            inputs=outputs,
            num_outputs=32,
            kernel_size=3,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training, trainable=trainable),
            trainable=trainable
        )
        outputs = slim.dropout(scope='DO_2', inputs=outputs, keep_prob=.8, is_training=is_training)

        outputs = slim.conv2d(
            scope='C_3',
            inputs=outputs,
            num_outputs=32,
            kernel_size=3,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training, trainable=trainable),
            trainable=trainable
        )
        outputs = slim.dropout(scope='DO_3', inputs=outputs, keep_prob=.8, is_training=is_training)

        outputs = slim.conv2d(
            scope='DC_4',
            inputs=outputs,
            num_outputs=256,
            kernel_size=[1, 97],
            rate=[1, 3],
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training, trainable=trainable),
            trainable=trainable
        )
        outputs = outputs[:, :, : 88 * 3, :]
        outputs.set_shape([None, None, 88 * 3, 256])
        outputs = slim.max_pool2d(scope='MP_4', inputs=outputs, kernel_size=[1, 3], stride=[1, 3], padding='VALID')
        outputs.set_shape([None, None, 88, 256])
        outputs = slim.dropout(scope='DO_4', inputs=outputs, keep_prob=.8, is_training=is_training)

        outputs = slim.fully_connected(
            scope='FC_5',
            inputs=outputs,
            num_outputs=64,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training, trainable=trainable),
            trainable=trainable
        )
        outputs = slim.dropout(scope='DO_5', inputs=outputs, keep_prob=.8, is_training=is_training)
        outputs.set_shape([None, None, 88, 64])

        if not use_feature:
            outputs = slim.fully_connected(
                scope='FC_6',
                inputs=outputs,
                num_outputs=1,
                activation_fn=None,
                trainable=trainable
            )
            outputs = tf.squeeze(outputs, axis=-1)
            outputs.set_shape([None, None, 88])

        return outputs

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
    def split_train_valid_and_test_files_fn(dataset):
        assert dataset in ('maps', 'maestro')
        if dataset == 'maps':
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
        else:
            year_name_split_list = MiscFns.get_maestro_year_name_split_list_fn()
            split_dict = collections.defaultdict(list)
            for year, rec_name, split in year_name_split_list:
                assert split in ('train', 'validation', 'test')
                if split == 'train':
                    split_dict['training'].append([year, rec_name])
                else:
                    split_dict[split].append([year, rec_name])
            assert len(split_dict['training']) == 954
            assert len(split_dict['validation']) == 105
            assert len(split_dict['test']) == 125

            return split_dict

    @staticmethod
    def gen_split_list_fn(num_frames, snippet_len):
        split_frames = range(0, num_frames + 1, snippet_len)
        if split_frames[-1] != num_frames:
            split_frames.append(num_frames)
        start_end_frame_pairs = zip(split_frames[:-1], split_frames[1:])

        return start_end_frame_pairs

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
    def num_samples_to_num_frames_fn(num_samples):
        assert isinstance(num_samples, (int, long))
        num_frames = (num_samples + 63) // 64
        num_frames = (num_frames + 21) // 22

        return num_frames

    @staticmethod
    def array_to_table_tf_fn(tf_array, header, scope, title, names, precision=None):
        tf_array = tf.convert_to_tensor(tf_array)
        assert tf_array._rank() == 2
        num_examples = tf_array.shape[0].value
        num_fields = tf_array.shape[1].value
        assert num_examples is not None
        assert num_fields is not None
        assert isinstance(header, list)
        assert len(header) == num_fields
        header = ['id', 'name'] + header
        header = tf.constant(header)
        assert isinstance(names, list)
        assert len(names) == num_examples
        names = tf.constant(names)[:, None]
        assert names.dtype == tf.string
        ids = [str(i) for i in range(1, num_examples + 1)]
        ids = tf.constant(ids)[:, None]
        if precision is None:
            if tf_array.dtype in (tf.float32, tf.float64):
                precision = 4
            else:
                precision = -1
        tf_array = tf.as_string(tf_array, precision=precision)
        tf_array = tf.concat([ids, names, tf_array], axis=1)
        tf_array.set_shape([num_examples, num_fields + 2])
        tf_array = tf.strings.reduce_join(tf_array, axis=1, separator=' | ')
        tf_array = tf.strings.reduce_join(tf_array, separator='\n')
        header = tf.strings.reduce_join(header, separator=' | ')
        sep = tf.constant(['---'])
        sep = tf.tile(sep, [num_fields + 2])
        sep = tf.strings.reduce_join(sep, separator=' | ')
        tf_array = tf.strings.join([header, sep, tf_array], separator='\n')
        assert isinstance(title, str)
        tf_array = tf.strings.join([tf.constant(title), tf_array], separator='\n\n')
        assert isinstance(scope, str)
        op = tf.summary.text(scope, tf_array)

        return op

    @staticmethod
    def from_maps_am_or_cl_fn(wav_or_mid_file):
        p = re.compile('.+(_.+)$')
        s = os.path.basename(wav_or_mid_file)
        s = os.path.splitext(s)[0]
        m = p.match(s)
        assert m is not None
        m = m.group(1)
        assert m is not None
        if m == '_ENSTDkAm':
            return 'am'
        elif m == '_ENSTDkCl':
            return 'cl'
        else:
            return None

    @staticmethod
    def unstack_88_into_batch_dim_fn(note_dim, inputs):
        outputs = inputs
        input_dims = outputs._rank()
        assert outputs.shape[note_dim].value == 88
        outputs = tf.unstack(outputs, axis=note_dim)
        assert len(outputs) == 88
        outputs = tf.concat(outputs, axis=0)
        output_dims = outputs._rank()
        assert input_dims - output_dims == 1

        return outputs

    @staticmethod
    def split_batch_dim_into_88_fn(note_dim, inputs):
        outputs = inputs
        input_dims = outputs._rank()
        outputs = tf.split(value=outputs, num_or_size_splits=88, axis=0)
        assert len(outputs) == 88
        outputs = tf.stack(outputs, axis=note_dim)
        assert outputs.shape[note_dim].value == 88
        output_dims = outputs._rank()
        assert output_dims - input_dims == 1

        return outputs

    @staticmethod
    def rnn_layer_fn(inputs, trainable):
        outputs = inputs
        outputs.set_shape([1, None, 88, 64])
        lstm_cell = tf.nn.rnn_cell.LSTMCell(name='lstm_cell', num_units=64, dtype=tf.float32, trainable=trainable)
        outputs = MiscFns.unstack_88_into_batch_dim_fn(note_dim=2, inputs=outputs)
        outputs, _ = tf.nn.dynamic_rnn(
            scope='dy_rnn',
            cell=lstm_cell,
            inputs=outputs,
            dtype=tf.float32
        )
        outputs = MiscFns.split_batch_dim_into_88_fn(note_dim=2, inputs=outputs)
        outputs.set_shape([1, None, 88, 64])

        return outputs

    @staticmethod
    def onset_detector_fn(inputs, is_training):
        inputs.set_shape([1, None, 336])
        assert isinstance(is_training, bool)
        assert tf.get_variable_scope().name == ''

        with tf.variable_scope('onset_detector', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('cnn_layers'):
                cnn_features = MiscFns.acoustic_model_fn(
                    spec_batch=inputs,
                    is_training=is_training,
                    use_feature=True,
                    trainable=True
                )
                cnn_features.set_shape([1, None, 88, 64])

            with tf.variable_scope('rnn_layer'):
                rnn_features = MiscFns.rnn_layer_fn(inputs=cnn_features, trainable=True)
                rnn_features.set_shape([1, None, 88, 64])

            with tf.variable_scope('output_layer'):
                logits = slim.fully_connected(
                    scope='FC',
                    inputs=rnn_features,
                    num_outputs=1,
                    activation_fn=None,
                    trainable=True
                )
                logits = tf.squeeze(logits, axis=-1)
                logits.set_shape([1, None, 88])

        return logits

    @staticmethod
    def cal_tps_fps_tns_fns_tf_fn(pred, target):
        assert pred.dtype == tf.bool and target.dtype == tf.bool
        npred = tf.logical_not(pred)
        ntarget = tf.logical_not(target)
        tps = tf.logical_and(pred, target)
        fps = tf.logical_and(pred, ntarget)
        tns = tf.logical_and(npred, ntarget)
        fns = tf.logical_and(npred, target)
        tps, fps, tns, fns = [tf.reduce_sum(tf.cast(value, tf.int32)) for value in [tps, fps, tns, fns]]
        inc_tps_fps_tns_fns = tf.convert_to_tensor([tps, fps, tns, fns], dtype=tf.int64)

        return inc_tps_fps_tns_fns

    @staticmethod
    def maestro_vqt_sg_and_label_fn(year, rec_name, label_only):

        def _get_num_frames_fn(year, rec_name):
            wav_file = os.path.join(os.environ['maestro'], year, rec_name + '.wav')
            wav_info = soundfile.info(wav_file)
            assert wav_info.samplerate in (44100, 48000)
            sr = 44100
            if wav_info.samplerate == 48000:
                num_frames = (wav_info.frames * sr + wav_info.samplerate - 1) // wav_info.samplerate
            else:
                num_frames = wav_info.frames
            num_frames = MiscFns.num_samples_to_num_frames_fn(num_frames)

            return num_frames

        assert isinstance(label_only, bool)
        num_frames = _get_num_frames_fn(year, rec_name)

        if not label_only:
            vqt_file = os.path.join(os.environ['maestro_vqt'], year, rec_name + '.vqt')
            _rec_name, vqt = MiscFns.load_np_array_from_file_fn(vqt_file)
            assert _rec_name == rec_name
            assert vqt.shape == (num_frames, 336)
            assert vqt.dtype == np.float32

        sr = 44100
        mid_file = os.path.join(os.environ['maestro'], year, rec_name + '.midi')
        num_frames_from_midi = mido.MidiFile(mid_file).length
        num_frames_from_midi = int(np.ceil(num_frames_from_midi * sr))
        num_frames_from_midi = MiscFns.num_samples_to_num_frames_fn(num_frames_from_midi)
        num_frames_from_midi += 2
        num_frames = min(num_frames, num_frames_from_midi)

        if not label_only:
            vqt = vqt[:num_frames]
            vqt = np.require(vqt, dtype=np.float32, requirements=['O', 'C'])
            vqt.flags['WRITEABLE'] = False

        label = MiscFns.label_fn(
            sr=sr,
            mid_file_name=mid_file,
            num_frames=num_frames,
            spec_stride=22 * 64
        )
        assert label.shape == (num_frames, 88) and label.dtype == np.bool_
        label = np.require(label, dtype=np.bool_, requirements=['O', 'C'])
        label.flags['WRITEABLE'] = False

        if label_only:
            return label
        else:
            return dict(sg=vqt, label=label)

    @staticmethod
    def maps_vqt_sg_and_label_fn(wav_file, label_only):

        wav_info = soundfile.info(wav_file)
        assert wav_info.samplerate == 44100
        num_frames = MiscFns.num_samples_to_num_frames_fn(wav_info.frames)

        if not label_only:
            rec_name = os.path.basename(wav_file)[:-4]
            vqt_file = os.path.join(os.environ['maps_vqt'], rec_name + '.vqt')
            _rec_name, vqt = MiscFns.load_np_array_from_file_fn(vqt_file)
            assert _rec_name == rec_name
            assert vqt.shape == (num_frames, 336)
            assert vqt.dtype == np.float32

        mid_file = wav_file[:-3] + 'mid'
        num_frames_from_midi = mido.MidiFile(mid_file).length
        num_frames_from_midi = int(np.ceil(num_frames_from_midi * wav_info.samplerate))
        num_frames_from_midi = MiscFns.num_samples_to_num_frames_fn(num_frames_from_midi)
        num_frames_from_midi += 2
        num_frames = min(num_frames, num_frames_from_midi)

        if not label_only:
            vqt = vqt[:num_frames]
            vqt = np.require(vqt, dtype=np.float32, requirements=['O', 'C'])
            vqt.flags['WRITEABLE'] = False

        label = MiscFns.label_fn(
            sr=wav_info.samplerate,
            mid_file_name=mid_file,
            num_frames=num_frames,
            spec_stride=64 * 22
        )
        assert label.dtype == np.bool_
        label.flags['WRITEABLE'] = False

        if label_only:
            return label
        else:
            return dict(sg=vqt, label=label)

    @staticmethod
    def cal_prf_tf_fn(tps, fps, fns):
        assert tps.dtype == tf.float64
        p = tps / (tps + fps + 1e-7)
        r = tps / (tps + fns + 1e-7)
        f = 2. * p * r / (p + r + 1e-7)
        return p, r, f

    @staticmethod
    def onset_transcription_performance_fn(logits, note_seq_ref):

        def _find_peak_locations_fn(logits, size, threshold):
            assert logits.shape[1:] == (88,)
            logits_max = maximum_filter1d(logits, size=size, axis=0, mode='constant')
            assert logits_max.shape == logits.shape
            logits_peak_ids = np.logical_and(logits == logits_max, logits > threshold)
            logits_peak_ids = np.where(logits_peak_ids)

            return logits_peak_ids

        onsets_peak_picked = np.empty_like(logits, dtype=np.bool_)
        onsets_peak_picked.fill(False)
        onsets_peak_picked[_find_peak_locations_fn(logits, 5, 0.)] = True
        onsets_peak_picked[-2:] = False

        pred_intervals = []
        pred_pitches = []
        for pitch in xrange(88):
            onset_frame_indices = np.where(onsets_peak_picked[:, pitch])[0]
            if len(onset_frame_indices):
                onset_frame_indices = onset_frame_indices.astype(np.float64)
                start_times = onset_frame_indices * 64 * 22 / 44100
                end_times = (onset_frame_indices + 1.) * 64 * 22 / 44100
                intervals = np.stack([start_times, end_times], axis=1)
                pred_intervals.append(intervals)
                pitches = np.zeros(len(intervals), dtype=np.uint8)
                pitches.fill(pitch + 21)
                pred_pitches.append(pitches)
        if pred_intervals:
            pred_intervals = np.concatenate(pred_intervals, axis=0).reshape(-1, 2)
            pred_pitches = np.concatenate(pred_pitches).reshape(-1)
            assert len(pred_intervals) == len(pred_pitches)

            tmp = MiscFns.note_seq_to_valued_intervals(note_seq_ref)
            ref_intervals = tmp['times']
            ref_pitches = tmp['pitches']

            prf = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=librosa.midi_to_hz(ref_pitches),
                est_intervals=pred_intervals,
                est_pitches=librosa.midi_to_hz(pred_pitches),
                offset_ratio=None
            )[:3]
        else:
            prf = [0., 0., 0.]

        return prf

    @staticmethod
    def get_note_seq_from_mid_file_fn(mid_file_name):
        note_seq = magenta.music.midi_file_to_note_sequence(mid_file_name)
        note_seq = magenta.music.apply_sustain_control_changes(note_seq)

        return note_seq

    @staticmethod
    def note_seq_to_valued_intervals(note_seq):
        start_end_times = []
        pitches = []

        for note in note_seq.notes:
            assert 21 <= note.pitch <= 108
            start_end_times.append((note.start_time, note.end_time))
            pitches.append(note.pitch)

        start_end_times = np.asarray(start_end_times, dtype=np.float64).reshape(-1, 2)
        pitches = np.asarray(pitches, dtype=np.uint8)

        return dict(times=start_end_times, pitches=pitches)


# all configurations go here
class Config(object):

    def __init__(self):
        self.debug_mode = DEBUG
        self.gpu_id = GPU_ID

        self.training_dataset_is_maps = TRAINING_DATASET
        assert self.training_dataset_is_maps in ('maps', 'maestro')
        self.training_dataset_is_maps = self.training_dataset_is_maps == 'maps'

        self.prefetch_ratio = PREFETCH_RATIO
        assert 0 < self.prefetch_ratio <= 1

        self.snippet_len = 600
        if self.training_dataset_is_maps:
            self.num_epochs = 40
            self.batches_per_epoch = 5000
        else:
            self.num_epochs = 45
            self.batches_per_epoch = 10000

        self.learning_rate = 1e-4

        """
        Instructions on configuring the running mode of this code

        1. The code can run in inference mode or training mode.
        2. If you want to run in inference mode, set variable inference to point to a saved model, e.g., inference = 
           os.path.join('saved_model', 'd0_epoch_8_of_15'), and set variables from_saved and model_prefix both to None.
        3. If you want to run in training mode, set variable inference to None and set variable from_saved to 
           point to a saved model if you want to continue training from the saved model or otherwise set from_saved to 
           None to train from scratch.
        4. In training mode, if you want to save the trained models, you can specify a prefix by setting variable 
           model_prefix. In training mode, if variable model_prefix is None, the trained models will not be saved.
        5. All statistics are saved as tensorboard summaries. Set variable tb_dir to naming the folder for storing 
           the data for tensorboard.
        """
        self.train_or_inference = Namespace(
            inference=None,
            from_saved=None,
            model_prefix='d0'
        )
        self.tb_dir = 'tb_d0'

        # If in non-debug mode, check if tb_dir and models (checkpoints) with the same prefix exist.
        # In non-debug mode, folder tb_dir cannot exist beforehand, because the the data in this folder may be overwritten
        if not self.debug_mode:
            # check if tb_dir exists
            assert self.tb_dir is not None
            tmp_dirs = glob.glob('./*/')
            tmp_dirs = [s[2:-1] for s in tmp_dirs]
            assert self.tb_dir not in tmp_dirs

            # check if model exists
            if self.train_or_inference.inference is None and self.train_or_inference.model_prefix is not None:
                if os.path.isdir('./saved_model'):
                    tmp_prefixes = glob.glob('./saved_model/*')
                    prog = re.compile(r'./saved_model/(.+?)_')
                    tmp = []
                    for file_name in tmp_prefixes:
                        try:
                            prefix = prog.match(file_name).group(1)
                        except AttributeError:
                            pass
                        else:
                            tmp.append(prefix)
                    tmp_prefixes = set(tmp)
                    assert self.train_or_inference.model_prefix not in tmp_prefixes

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.gpu_config = gpu_config

        # get splits for training, validation and test
        split_maestro = MiscFns.split_train_valid_and_test_files_fn('maestro')
        split_maps = MiscFns.split_train_valid_and_test_files_fn('maps')
        tvt_split_dict = dict(
            test_maps=split_maps['test'],
            test_maestro=split_maestro['test'],
            validation_maps=split_maps['validation'],
            validation_maestro=split_maestro['validation']
        )
        if self.training_dataset_is_maps:
            tvt_split_dict['training_maps'] = split_maps['training']
        else:
            tvt_split_dict['training_maestro'] = split_maestro['training']
        self.tvt_split_dict = tvt_split_dict

        # make sure the first split is the training split.
        self.model_names = self.tvt_split_dict.keys()
        if 'training' not in self.model_names[0]:
            for idx, name in enumerate(self.model_names):
                if 'training' in name:
                    break
            else:
                assert False
            self.model_names[0], self.model_names[idx] = self.model_names[idx], self.model_names[0]
            assert len(self.model_names) == 5

        # use few recordings if in debug mode
        if self.debug_mode:
            np.random.seed(100)
            for tvt in self.tvt_split_dict.keys():
                _num = len(self.tvt_split_dict[tvt])
                _sel = np.random.choice(_num, 5, replace=False)
                self.tvt_split_dict[tvt] = [self.tvt_split_dict[tvt][ii] for ii in _sel]

            self.num_epochs = 4
            self.batches_per_epoch = 200
            if self.prefetch_ratio < .999:
                self.prefetch_ratio = .5

        # use only one recording for training and validation if in inference mode
        if self.train_or_inference.inference is not None:
            for model_name in self.model_names:
                if 'training' in model_name or 'validation' in model_name:
                    del self.tvt_split_dict[model_name][1:]


# define neural network models
class Model(object):
    def __init__(self, config, name):
        assert name in config.model_names
        self.name = name
        self.is_training = True if 'training' in self.name else False
        self.config = config
        self._gen_dataset_fn()
        self.batch = self._gen_batch_fn()

        with tf.name_scope(self.name):
            logits = self._nn_model_fn()
            logits.set_shape([1, None, 88])
            self.logits = logits
            self.loss = self._loss_fn()
            if self.is_training:
                self.training_op = self._training_op_fn()
            self.stats = self._stats_fn()
            self.tb_proto = self._tb_summary_fn()
            if not self.is_training:
                self.peak_picked_pl_and_tb_proto = self._tb_proto_for_peak_picked_detections_fn()

    def _dataset_iter_fn(self):
        if self.is_training:
            if self.config.prefetch_ratio < .999:
                if self.config.training_dataset_is_maps:

                    def _read_vqt_fn(rec_name):
                        vqt_file = os.path.join(os.environ['maps_vqt'], rec_name + '.vqt')
                        _rec_name, vqt = MiscFns.load_np_array_from_file_fn(vqt_file)
                        assert _rec_name == rec_name
                        assert vqt.dtype == np.float32 and vqt.shape[1:] == (336,)

                        return vqt

                    logging.info('{} - enter generator'.format(self.name))
                    assert hasattr(self, 'dataset')
                    wav_file_list = self.config.tvt_split_dict[self.name]
                    num_recs = len(wav_file_list)
                    num_frames_list = [len(label_dict['label']) for label_dict in self.dataset]
                    n_iters = 0
                    shuffled_rec_idx_list = range(num_recs)
                    while True:
                        np.random.shuffle(shuffled_rec_idx_list)
                        for ord_idx, rec_idx in enumerate(shuffled_rec_idx_list):
                            wav_file = wav_file_list[rec_idx]
                            rec_name = os.path.basename(wav_file)[:-4]
                            if n_iters == 0:
                                logging.info('{}/{} - {} - reading vqt'.format(ord_idx + 1, num_recs, rec_name))
                            vqt = _read_vqt_fn(rec_name)
                            num_frames = num_frames_list[rec_idx]
                            assert vqt.shape[0] >= num_frames
                            vqt = vqt[:num_frames]
                            assert vqt.shape == (num_frames, 336)
                            vqt = np.require(vqt, dtype=np.float32, requirements=['O', 'C'])
                            vqt.flags['WRITEABLE'] = False

                            label_dict = self.dataset[rec_idx]
                            assert label_dict['rec_name'] == rec_name
                            label = label_dict['label']
                            assert len(label) == num_frames
                            split_list = []
                            split_list[:] = label_dict['split_list']
                            np.random.shuffle(split_list)
                            for start, end in split_list:
                                yield dict(
                                    spectrogram=vqt[start:end],
                                    label=label[start:end],
                                    num_frames=end - start
                                )
                        n_iters += 1
                        logging.info('{} iters over {} done'.format(n_iters, self.name))
                else:  # training dataset is maestro
                    def _read_vqt_fn(year, rec_name):
                        vqt_file = os.path.join(os.environ['maestro_vqt'], year, rec_name + '.vqt')
                        _rec_name, vqt = MiscFns.load_np_array_from_file_fn(vqt_file)
                        assert _rec_name == rec_name
                        assert vqt.dtype == np.float32 and vqt.shape[1:] == (336,)

                        return vqt

                    logging.info('{} - enter generator'.format(self.name))
                    assert hasattr(self, 'dataset')
                    year_name_list = self.config.tvt_split_dict[self.name]
                    num_recs = len(year_name_list)
                    num_frames_list = [len(label_dict['label']) for label_dict in self.dataset]
                    num_iters = 0
                    shuffled_rec_idx_list = range(num_recs)
                    while True:
                        np.random.shuffle(shuffled_rec_idx_list)
                        for ord_idx, rec_idx in enumerate(shuffled_rec_idx_list):
                            year, rec_name = year_name_list[rec_idx]
                            if num_iters == 0:
                                logging.info('{}/{} - {} - reading vqt'.format(ord_idx + 1, num_recs, rec_name))
                            vqt = _read_vqt_fn(year, rec_name)
                            num_frames = num_frames_list[rec_idx]
                            assert vqt.shape[0] >= num_frames
                            vqt = vqt[:num_frames]
                            assert vqt.shape == (num_frames, 336)
                            vqt = np.require(vqt, dtype=np.float32, requirements=['O', 'C'])
                            vqt.flags['WRITEABLE'] = False

                            label_dict = self.dataset[rec_idx]
                            assert label_dict['rec_name'] == rec_name
                            label = label_dict['label']
                            assert len(label) == num_frames
                            split_list = []
                            split_list[:] = label_dict['split_list']
                            np.random.shuffle(split_list)
                            for start, end in split_list:
                                yield dict(
                                    spectrogram=vqt[start:end],
                                    label=label[start:end],
                                    num_frames=end - start
                                )
                        num_iters += 1
                        logging.info('{} iters over {} done'.format(num_iters, self.name))
            else:  # vqts have been read
                logging.info('{} - enter generator'.format(self.name))
                n_iters = 0
                while True:
                    np.random.shuffle(self.rec_start_end_list)
                    for rec_idx, start, end in self.rec_start_end_list:
                        rec_dict = self.dataset[rec_idx]
                        yield dict(
                            spectrogram=rec_dict['sg'][start:end],
                            label=rec_dict['label'][start:end],
                            num_frames=end - start
                        )
                    n_iters += 1
                    logging.info('{} iters over the training split done'.format(n_iters))

        if not self.is_training:
            logging.debug('{} - enter generator'.format(self.name))
            assert hasattr(self, 'dataset')

            logging.debug('{} - generator begins'.format(self.name))
            for rec_idx, rec_dict in enumerate(self.dataset):
                split_list = rec_dict['split_list']
                for start_frame, end_frame in split_list:
                    yield dict(
                        spectrogram=rec_dict['sg'][start_frame:end_frame],
                        label=rec_dict['label'][start_frame:end_frame],
                        num_frames=end_frame - start_frame,
                        rec_idx=rec_idx
                    )
            logging.debug('{} - generator ended'.format(self.name))

    def _gen_batch_fn(self):
        with tf.device('/cpu:0'):
            if self.is_training:
                dataset = tf.data.Dataset.from_generator(
                    generator=self._dataset_iter_fn,
                    output_types=dict(spectrogram=tf.float32, label=tf.bool, num_frames=tf.int32),
                    output_shapes=dict(spectrogram=[None, 336], label=[None, 88], num_frames=[])
                )
                dataset = dataset.batch(1)

                if self.config.prefetch_ratio < .999:
                    num_pres = int(self.total_num_snippets * self.config.prefetch_ratio)
                    logging.info('training - prefetched snippets - {}'.format(num_pres))
                else:
                    num_pres = 5
                dataset = dataset.prefetch(num_pres)
                dataset_iter = dataset.make_one_shot_iterator()
                element = dataset_iter.get_next()

                return element
            else:  # not self.is_training
                _sg_shape = [None, 336]
                dataset = tf.data.Dataset.from_generator(
                    generator=self._dataset_iter_fn,
                    output_types=dict(spectrogram=tf.float32, label=tf.bool, num_frames=tf.int32, rec_idx=tf.int32),
                    output_shapes=dict(spectrogram=_sg_shape, label=[None, 88], num_frames=[], rec_idx=[])
                )
                dataset = dataset.batch(1)
                dataset = dataset.prefetch(5)
                self.reinitializable_iter_for_dataset = dataset.make_initializable_iterator()
                element = self.reinitializable_iter_for_dataset.get_next()
                element['spectrogram'].set_shape([1] + _sg_shape)
                element['label'].set_shape([1, None, 88])
                element['num_frames'].set_shape([1])
                element['rec_idx'].set_shape([1])

                return element

    def _nn_model_fn(self):
        inputs = self.batch['spectrogram']
        _nn_fn = MiscFns.onset_detector_fn
        inputs.set_shape([1, None, 336])
        outputs = _nn_fn(inputs=inputs, is_training=self.is_training)
        outputs.set_shape([1, None, 88])

        return outputs

    def _gen_dataset_fn(self):
        if self.is_training:
            assert not hasattr(self, 'dataset')
            file_names = self.config.tvt_split_dict[self.name]
            num_recs = len(file_names)
            label_only = self.config.prefetch_ratio < .999
            if label_only:
                logging.info('{} - generate labels'.format(self.name))
            else:
                logging.info('{} - generate spectrograms and labels'.format(self.name))
            dataset = []

            if self.config.training_dataset_is_maps:
                for file_idx, wav_file in enumerate(file_names):
                    rec_name = os.path.basename(wav_file)[:-4]
                    logging.info('{}/{} - {}'.format(file_idx + 1, num_recs, rec_name))
                    sg_label = self._get_sg_and_label_fn(wav_file=wav_file)
                    if label_only:
                        assert isinstance(sg_label, np.ndarray)
                        assert not sg_label.flags['WRITEABLE']
                        assert sg_label.dtype == np.bool_ and sg_label.shape[1:] == (88,)
                        dataset.append(dict(rec_name=rec_name, label=sg_label))
                    else:
                        assert isinstance(sg_label, dict)
                        sg = sg_label['sg']
                        assert not sg.flags['WRITEABLE']
                        assert sg.dtype == np.float32 and sg.shape[1:] == (336,)

                        label = sg_label['label']
                        assert not label.flags['WRITEABLE']
                        assert label.dtype == np.bool_ and label.shape[1:] == (88,)

                        assert len(sg) == len(label)
                        dataset.append(dict(sg=sg, label=label))
            else:  # the dataset is maestro
                for file_idx, (year, rec_name) in enumerate(file_names):
                    logging.info('{}/{} - {}'.format(file_idx + 1, num_recs, rec_name))
                    sg_label = self._get_sg_and_label_fn(year=year, rec_name=rec_name)
                    if label_only:
                        assert isinstance(sg_label, np.ndarray)
                        assert not sg_label.flags['WRITEABLE']
                        assert sg_label.dtype == np.bool_ and sg_label.shape[1:] == (88,)
                        dataset.append(dict(rec_name=rec_name, label=sg_label))
                    else:
                        assert isinstance(sg_label, dict)
                        sg = sg_label['sg']
                        assert not sg.flags['WRITEABLE']
                        assert sg.dtype == np.float32
                        assert sg.shape[1:] == (336, )

                        label = sg_label['label']
                        assert not label.flags['WRITEABLE']
                        assert label.dtype == np.bool_ and label.shape[1:] == (88,)

                        assert len(label) == len(sg)
                        dataset.append(dict(sg=sg, label=label))

            self.dataset = dataset

            if label_only:
                total_num_splits = 0
                for rec_idx, rec_dict in enumerate(self.dataset):
                    split_list = MiscFns.gen_split_list_fn(
                        num_frames=len(rec_dict['label']),
                        snippet_len=self.config.snippet_len
                    )
                    rec_dict['split_list'] = split_list
                    total_num_splits += len(split_list)
                self.total_num_snippets = total_num_splits
                logging.info('{} - number of snippets per iteration over the dataset - {}'.format(self.name, total_num_splits))
            else:
                rec_start_end_list = []
                for rec_idx in xrange(num_recs):
                    split_list = MiscFns.gen_split_list_fn(
                        num_frames=len(self.dataset[rec_idx]['sg']),
                        snippet_len=self.config.snippet_len
                    )
                    l = [[rec_idx, s[0], s[1]] for s in split_list]
                    rec_start_end_list.extend(l)
                self.rec_start_end_list = rec_start_end_list
                logging.info('{} - number of snippets per iteration over the dataset - {}'.format(self.name, len(self.rec_start_end_list)))

        if not self.is_training:
            assert not hasattr(self, 'dataset')
            file_names = self.config.tvt_split_dict[self.name]
            num_recs = len(file_names)
            logging.info('{} - generate spectrograms and labels'.format(self.name))

            _, dataset_name = self.name.split('_')
            assert dataset_name in ('maps', 'maestro')
            if dataset_name == 'maps':
                dataset = []
                rec_names = []
                for file_idx, wav_file_name in enumerate(file_names):
                    rec_name = os.path.basename(wav_file_name)[:-4]
                    rec_names.append(rec_name)
                    logging.info('{}/{} - {}'.format(file_idx + 1, num_recs, rec_name))
                    sg_label_dict = self._get_sg_and_label_fn(wav_file=wav_file_name)
                    assert isinstance(sg_label_dict, dict)
                    sg = sg_label_dict['sg']
                    assert not sg.flags['WRITEABLE']
                    assert sg.dtype == np.float32 and sg.shape[1:] == (336,)

                    label = sg_label_dict['label']
                    assert not label.flags['WRITEABLE']
                    assert label.dtype == np.bool_ and label.shape[1:] == (88,)

                    assert len(sg) == len(label)

                    mid_file = wav_file_name[:-3] + 'mid'
                    note_seq = MiscFns.get_note_seq_from_mid_file_fn(mid_file)

                    dataset.append(dict(sg=sg, label=label, note_seq=note_seq))
                self.dataset = dataset
                self.rec_names = tuple(rec_names)
            else:
                dataset = []
                rec_names = []
                for file_idx, (year, rec_name) in enumerate(file_names):
                    rec_names.append(rec_name)
                    logging.info('{}/{} - {}'.format(file_idx + 1, num_recs, rec_name))
                    sg_label_dict = self._get_sg_and_label_fn(year=year, rec_name=rec_name)
                    assert isinstance(sg_label_dict, dict)
                    sg = sg_label_dict['sg']
                    assert not sg.flags['WRITEABLE']
                    assert sg.dtype == np.float32
                    assert sg.shape[1:] == (336,)

                    label = sg_label_dict['label']
                    assert not label.flags['WRITEABLE']
                    assert label.dtype == np.bool_ and label.shape[1:] == (88,)

                    assert len(sg) == len(label)

                    mid_file = os.path.join(os.environ['maestro'], year, rec_name + '.midi')
                    note_seq = MiscFns.get_note_seq_from_mid_file_fn(mid_file)

                    dataset.append(dict(sg=sg, label=label, note_seq=note_seq))
                self.dataset = dataset
                self.rec_names = tuple(rec_names)

            self.num_frames_vector = np.asarray([len(rec_dict['sg']) for rec_dict in self.dataset], dtype=np.int64)

            for rec_dict in self.dataset:
                split_list = MiscFns.gen_split_list_fn(
                    num_frames=len(rec_dict['sg']),
                    snippet_len=self.config.snippet_len
                )
                rec_dict['split_list'] = split_list

    def _loss_fn(self):
        logits = self.logits
        logits.set_shape([1, None, 88])
        logits = tf.squeeze(logits, axis=0)
        labels = self.batch['label']
        labels.set_shape([1, None, 88])
        assert labels.dtype == tf.bool
        labels = tf.squeeze(labels, axis=0)
        labels = tf.cast(labels, tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)

        return loss

    def _training_op_fn(self):
        loss = self.loss
        if self.is_training:
            _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if _update_ops:
                with tf.control_dependencies(_update_ops):
                    training_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)
            else:
                training_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)
            return training_op
        else:
            return None

    def _stats_fn(self):

        if not self.is_training:
            assert tf.get_variable_scope().name == ''
            num_recs = len(self.dataset)
            with tf.variable_scope(self.name):
                with tf.variable_scope('statistics'):
                    var_int64_ind_tps_fps_tns_fns = tf.get_variable(
                        name='var_int64_ind_tps_fps_tns_fns',
                        dtype=tf.int64,
                        shape=[num_recs, 4],
                        initializer=tf.zeros_initializer,
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )
                    var_float64_acc_loss = tf.get_variable(
                        name='var_float64_acc_loss',
                        dtype=tf.float64,
                        shape=[],
                        initializer=tf.zeros_initializer,
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )
                    var_int64_batch_counter = tf.get_variable(
                        name='var_int64_batch_counter',
                        dtype=tf.int64,
                        shape=[],
                        initializer=tf.zeros_initializer,
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )

                    pred_labels = tf.greater(tf.squeeze(self.logits, axis=0), 0.)
                    pred_labels.set_shape([None, 88])
                    target_labels = self.batch['label']
                    assert target_labels.dtype == tf.bool
                    target_labels.set_shape([1, None, 88])
                    target_labels = tf.squeeze(target_labels, axis=0)
                    inc = MiscFns.cal_tps_fps_tns_fns_tf_fn(
                        pred=pred_labels,
                        target=target_labels
                    )
                    assert inc.dtype == tf.int64
                    assert inc._rank() == 1

                    num_labels = tf.cast(self.batch['num_frames'][0], tf.int64) * tf.constant(88, dtype=tf.int64)
                    _num_labels = tf.reduce_sum(inc)
                    _assert_op = tf.assert_equal(num_labels, _num_labels)
                    with tf.control_dependencies([_assert_op]):
                        ind_update_op = tf.scatter_add(
                            var_int64_ind_tps_fps_tns_fns,
                            self.batch['rec_idx'][0],
                            inc
                        )

                    acc_loss_update_op = tf.assign_add(var_float64_acc_loss, tf.cast(self.loss, tf.float64))
                    batch_counter_update_op = tf.assign_add(var_int64_batch_counter, tf.constant(1, dtype=tf.int64))

                    # ind and average stats
                    tps, fps, _, fns = tf.unstack(tf.cast(var_int64_ind_tps_fps_tns_fns, tf.float64), axis=1)
                    ps, rs, fs = MiscFns.cal_prf_tf_fn(tps=tps, fps=fps, fns=fns)
                    ind_prfs = tf.stack([ps, rs, fs], axis=1)
                    ind_prfs.set_shape([num_recs, 3])
                    _num_labels = self.num_frames_vector
                    assert isinstance(_num_labels, np.ndarray) and _num_labels.dtype == np.int64
                    _num_labels = tf.constant(_num_labels) * tf.constant(88, dtype=tf.int64)
                    _num_labels_p = tf.reduce_sum(var_int64_ind_tps_fps_tns_fns, axis=1)
                    _assert_op = tf.assert_equal(_num_labels, _num_labels_p)
                    with tf.control_dependencies([_assert_op]):
                        ave_prf = tf.reduce_mean(ind_prfs, axis=0)

                    # ensemble stats
                    ensemble = tf.reduce_sum(var_int64_ind_tps_fps_tns_fns, axis=0)
                    tps, fps, _, fns = tf.unstack(tf.cast(ensemble, tf.float64))
                    p, r, f = MiscFns.cal_prf_tf_fn(tps=tps, fps=fps, fns=fns)
                    ave_loss = var_float64_acc_loss / tf.cast(var_int64_batch_counter, tf.float64)
                    with tf.control_dependencies([_assert_op]):
                        en_prf_and_loss = tf.convert_to_tensor([p, r, f, ave_loss])

                    update_op = tf.group(ind_update_op, batch_counter_update_op, acc_loss_update_op)

                    stats = dict(
                        individual_tps_fps_tns_fns=var_int64_ind_tps_fps_tns_fns,
                        individual_prfs=ind_prfs,
                        average_prf=ave_prf,
                        ensemble_tps_fps_tns_fns=ensemble,
                        ensemble_prf_and_loss=en_prf_and_loss
                    )
            return dict(update_op=update_op, value=stats)

        if self.is_training:
            logits = self.logits
            logits.set_shape([1, None, 88])
            loss = self.loss
            with tf.variable_scope(self.name):
                with tf.variable_scope('statistics'):
                    var_int64_tps_fps_tns_fns = tf.get_variable(
                        name='var_int64_tps_fps_tns_fns',
                        dtype=tf.int64,
                        shape=[4],
                        trainable=False,
                        initializer=tf.zeros_initializer,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )

                    var_float64_acc_loss = tf.get_variable(
                        name='var_float64_acc_loss',
                        dtype=tf.float64,
                        shape=[],
                        trainable=False,
                        initializer=tf.zeros_initializer,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )

                    var_int64_batch_counter = tf.get_variable(
                        name='var_int64_batch_counter',
                        dtype=tf.int64,
                        shape=[],
                        trainable=False,
                        initializer=tf.zeros_initializer,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )

                    pred_labels_flattened = tf.greater(tf.squeeze(logits, axis=0), 0.)

                    target_labels_flattened = self.batch['label']
                    target_labels_flattened = tf.squeeze(target_labels_flattened, axis=0)
                    inc_tps_fps_tns_fns = MiscFns.cal_tps_fps_tns_fns_tf_fn(
                        pred=pred_labels_flattened,
                        target=target_labels_flattened
                    )
                    assert inc_tps_fps_tns_fns.dtype == tf.int64
                    num_labels = self.batch['num_frames'][0]
                    num_labels = tf.cast(num_labels, tf.int64) * tf.constant(88, tf.int64)
                    _num_labels = tf.reduce_sum(inc_tps_fps_tns_fns)
                    _assert_op = tf.assert_equal(num_labels, _num_labels)
                    with tf.control_dependencies([_assert_op]):
                        ensemble_tps_fps_tns_fns_update_op = tf.assign_add(
                            var_int64_tps_fps_tns_fns,
                            inc_tps_fps_tns_fns
                        )

                    acc_loss_update_op = tf.assign_add(var_float64_acc_loss, tf.cast(loss, tf.float64))
                    batch_counter_update_op = tf.assign_add(var_int64_batch_counter, tf.constant(1, tf.int64))

                    en_tps_fps_tns_fns_float64 = tf.cast(var_int64_tps_fps_tns_fns, tf.float64)
                    tps, fps, _, fns = tf.unstack(en_tps_fps_tns_fns_float64)
                    p, r, f = MiscFns.cal_prf_tf_fn(tps=tps, fps=fps, fns=fns)
                    ave_loss = var_float64_acc_loss / tf.cast(var_int64_batch_counter, tf.float64)
                    prf_and_ave_loss = tf.convert_to_tensor([p, r, f, ave_loss])

                    update_op = tf.group(ensemble_tps_fps_tns_fns_update_op, acc_loss_update_op, batch_counter_update_op)
                    stats = dict(tps_fps_tns_fns=var_int64_tps_fps_tns_fns, prf_and_loss=prf_and_ave_loss)

            return dict(update_op=update_op, value=stats)

    def _tb_summary_fn(self):

        if self.is_training:
            scalar_summaries = []
            with tf.name_scope('statistics'):
                stats = self.stats['value']
                p, r, f, l = tf.unstack(stats['prf_and_loss'])
                summary_dict = dict(precision=p, recall=r, f1=f, loss=l)
                for sum_name, sum_value in summary_dict.iteritems():
                    scalar_summaries.append(tf.summary.scalar(sum_name, sum_value))
                scalar_summaries = tf.summary.merge(scalar_summaries)
            return scalar_summaries

        if not self.is_training:
            num_recs = len(self.dataset)
            tb_table_and_scalar_protos = []
            with tf.name_scope('statistics'):
                stats = self.stats['value']
                ind_prfs = stats['individual_prfs']
                ave_prf = stats['average_prf']
                en_prf, ave_loss = tf.split(stats['ensemble_prf_and_loss'], [3, 1])
                ave_loss = ave_loss[0]

                assert ind_prfs.dtype == ave_prf.dtype == en_prf.dtype == tf.float64
                prfs = tf.concat([ind_prfs, ave_prf[None, :], en_prf[None, :]], axis=0)
                prfs.set_shape([num_recs + 2, 3])
                names = list(self.rec_names) + ['average', 'ensemble']
                des = 'individual_and_their_average_and_ensemble_prfs'
                prf_tb_table_proto = MiscFns.array_to_table_tf_fn(
                    tf_array=prfs,
                    header=['p', 'r', 'f1'],
                    scope=des,
                    title=des.replace('_', ' '),
                    names=names
                )
                tb_table_and_scalar_protos.append(prf_tb_table_proto)

                ind_tps_fps_tns_fns = stats['individual_tps_fps_tns_fns']
                en_tps_fps_tns_fns = stats['ensemble_tps_fps_tns_fns']
                en_tps_fps_tns_fns = en_tps_fps_tns_fns[None, :]
                assert ind_tps_fps_tns_fns.dtype.base_dtype == en_tps_fps_tns_fns.dtype == tf.int64
                ind_en = tf.concat([ind_tps_fps_tns_fns, en_tps_fps_tns_fns], axis=0)
                names = list(self.rec_names)
                names.append('ensemble')
                des = 'individual and their ensemble tps fps tns and fns'
                tps_fps_tns_fns_tb_table_proto = MiscFns.array_to_table_tf_fn(
                    tf_array=ind_en,
                    header=['TP', 'FP', 'TN', 'FN'],
                    scope=des.replace(' ', '_'),
                    title=des,
                    names=names
                )
                tb_table_and_scalar_protos.append(tps_fps_tns_fns_tb_table_proto)

                prf_loss_tb_scalar_proto = []
                for ave_or_en in ('average', 'ensemble'):
                    with tf.name_scope(ave_or_en):
                        p, r, f = tf.unstack(ave_prf if ave_or_en == 'average' else en_prf)
                        prfl_protos = []
                        items_for_summary = dict(precision=p, recall=r, f1=f)
                        for item_name, item_value in items_for_summary.iteritems():
                            prfl_protos.append(tf.summary.scalar(item_name, item_value))
                        if ave_or_en == 'average':
                            prfl_protos.append(tf.summary.scalar('loss', ave_loss))
                        prf_loss_tb_scalar_proto.append(tf.summary.merge(prfl_protos))
                prf_loss_tb_scalar_proto = tf.summary.merge(prf_loss_tb_scalar_proto)

                tb_table_and_scalar_protos.append(prf_loss_tb_scalar_proto)
                tb_table_and_scalar_protos = tf.summary.merge(tb_table_and_scalar_protos)

            return tb_table_and_scalar_protos

    def _get_sg_and_label_fn(self, **kwargs):
        if self.is_training:
            if self.config.training_dataset_is_maps:
                return MiscFns.maps_vqt_sg_and_label_fn(
                    wav_file=kwargs['wav_file'],
                    label_only=self.config.prefetch_ratio < .999
                )
            else:
                return MiscFns.maestro_vqt_sg_and_label_fn(
                    year=kwargs['year'], rec_name=kwargs['rec_name'], label_only=self.config.prefetch_ratio < .999)
        else:
            split_name, dataset_name = self.name.split('_')
            assert split_name in ('validation', 'test') and dataset_name in ('maps', 'maestro')
            if dataset_name == 'maps':
                sg_label_dict = MiscFns.maps_vqt_sg_and_label_fn(wav_file=kwargs['wav_file'], label_only=False)
            else:
                sg_label_dict = MiscFns.maestro_vqt_sg_and_label_fn(
                    year=kwargs['year'], rec_name=kwargs['rec_name'], label_only=False)

            return sg_label_dict

    def _tb_proto_for_peak_picked_detections_fn(self):
        assert not self.is_training
        _num_recs = len(self.dataset)
        with tf.name_scope('peak_picked_statistics'):
            prf_array_pl = tf.placeholder(dtype=tf.float32, shape=[_num_recs, 3])
            ave_prf_pl = tf.reduce_mean(prf_array_pl, axis=0)
            prf_array_w_ave = tf.concat([prf_array_pl, ave_prf_pl[None, :]], axis=0)
            names = list(self.rec_names) + ['average']
            tb_proto_ind_prfs_table = MiscFns.array_to_table_tf_fn(
                tf_array=prf_array_w_ave,
                header=['p', 'r', 'f'],
                scope='ind_prfs',
                title='individual and their average prfs',
                names=names
            )
            p, r, f = tf.unstack(ave_prf_pl)
            tmp = dict(ave_p=p, ave_r=r, ave_f=f)
            prf_tb_protos = []
            for name, data in tmp.iteritems():
                tb_scalar_proto = tf.summary.scalar(name, data)
                prf_tb_protos.append(tb_scalar_proto)
            prf_tb_protos = tf.summary.merge(prf_tb_protos)
            peak_picked_pl_and_tb_proto = dict(
                pl=prf_array_pl,
                tb_proto=tf.summary.merge([tb_proto_ind_prfs_table, prf_tb_protos])
            )

            return peak_picked_pl_and_tb_proto


def main():
    MODEL_DICT = {}
    MODEL_DICT['config'] = Config()
    for name in MODEL_DICT['config'].model_names:
        MODEL_DICT[name] = Model(config=MODEL_DICT['config'], name=name)

    aug_info_pl = tf.placeholder(dtype=tf.string, name='aug_info_pl')
    aug_info_summary = tf.summary.text('aug_info_summary', aug_info_pl)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(MODEL_DICT['config'].gpu_id)
    with tf.Session(config=MODEL_DICT['config'].gpu_config) as sess:
        # summary writer
        summary_writer_dict = {}
        for model_name in MODEL_DICT['config'].model_names:
            summary_writer_dict[model_name] = tf.summary.FileWriter(
                os.path.join(MODEL_DICT['config'].tb_dir, model_name))

        aug_info = []
        aug_info.append('onset detection')
        training_dataset = MODEL_DICT['config'].training_dataset_is_maps
        training_dataset = 'maps' if training_dataset else 'maestro'
        aug_info.append('onset detection - training dataset - {}'.format(training_dataset))
        aug_info.append('training data prefetch ratio - {}'.format(MODEL_DICT['config'].prefetch_ratio))

        if MODEL_DICT['config'].train_or_inference.inference is not None:
            aug_info.append('inference with {}'.format(MODEL_DICT['config'].train_or_inference.inference))
        elif MODEL_DICT['config'].train_or_inference.from_saved is not None:
            aug_info.append('continue training from {}'.format(MODEL_DICT['config'].train_or_inference.from_saved))

        if MODEL_DICT['config'].train_or_inference.inference is None:
            _model_prefix = MODEL_DICT['config'].train_or_inference.model_prefix
            if _model_prefix is not None:
                aug_info.append('model prefix {}'.format(_model_prefix))
            else:
                aug_info.append('model will not be saved')

        aug_info.append('tb dir - {}'.format(MODEL_DICT['config'].tb_dir))
        aug_info.append('debug mode - {}'.format(MODEL_DICT['config'].debug_mode))
        aug_info.append('snippet length - {}'.format(MODEL_DICT['config'].snippet_len))
        aug_info.append('batch size - 1')
        aug_info.append('num of batches per epoch - {}'.format(MODEL_DICT['config'].batches_per_epoch))
        aug_info.append('num of epochs - {}'.format(MODEL_DICT['config'].num_epochs))
        aug_info.append('training start time - {}'.format(datetime.datetime.now()))
        aug_info = '\n\n'.join(aug_info)
        logging.info(aug_info)
        summary_writer_dict[MODEL_DICT['config'].model_names[0]].add_summary(sess.run(aug_info_summary, feed_dict={aug_info_pl: aug_info}))

        logging.info('local vars -')
        for idx, var in enumerate(tf.local_variables()):
            logging.info('{}\t{}'.format(idx, var.op.name))

        logging.info('trainable vars -')
        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
            logging.info('{}\t{}\t{}'.format(idx, var.op.name, var.shape))

        OP_DICT = {}
        for model_name in MODEL_DICT['config'].model_names:
            m = MODEL_DICT[model_name]

            if m.is_training:
                tmp = dict(
                    batch=[m.training_op, m.stats['update_op']],
                    epoch=m.tb_proto
                )
            else:
                tmp = dict(
                    batch=dict(
                        rec_idx=m.batch['rec_idx'],
                        logits=m.logits,
                        update_op=m.stats['update_op']
                    ),
                    epoch=m.tb_proto
                )

            OP_DICT[model_name] = tmp

        def test_or_validate_fn(valid_or_test, global_step=None):
            assert valid_or_test in MODEL_DICT['config'].model_names and 'training' not in valid_or_test

            ops_per_batch = OP_DICT[valid_or_test]['batch']
            ops_per_epoch = OP_DICT[valid_or_test]['epoch']

            batch_idx = 0
            _dataset_test = MODEL_DICT[valid_or_test].dataset
            total_num_snippets = sum(len(rec_dict['split_list']) for rec_dict in _dataset_test)
            num_recs = len(_dataset_test)

            onset_prfs = np.empty((num_recs, 3), dtype=np.float32)
            for rec_idx in xrange(num_recs):
                rec_dict = _dataset_test[rec_idx]
                split_list = rec_dict['split_list']
                num_snippets = len(split_list)
                num_frames = len(rec_dict['sg'])
                assert num_frames == MODEL_DICT[valid_or_test].num_frames_vector[rec_idx]
                onset_logits = []
                for snippet_idx in xrange(num_snippets):
                    logging.debug('batch {}/{}'.format(batch_idx + 1, total_num_snippets))
                    tmp = sess.run(ops_per_batch)
                    _rec_idx = tmp['rec_idx'][0]
                    assert _rec_idx == rec_idx
                    logits = tmp['logits']
                    assert logits.shape == (1, split_list[snippet_idx][1] - split_list[snippet_idx][0], 88)
                    onset_logits.append(np.squeeze(logits, axis=0))
                    batch_idx += 1
                onset_logits = np.concatenate(onset_logits, axis=0)
                assert onset_logits.shape == (num_frames, 88)
                prf = MiscFns.onset_transcription_performance_fn(onset_logits, rec_dict['note_seq'])
                onset_prfs[rec_idx] = prf
                logging.info('{}/{} - {}'.format(rec_idx + 1, num_recs, MODEL_DICT[valid_or_test].rec_names[rec_idx]))
                logging.info('peak-picked onset performance - {}'.format(prf))
            summary_writer_dict[valid_or_test].add_summary(sess.run(ops_per_epoch), global_step)
            ave_prf = np.mean(onset_prfs, axis=0)
            logging.info('peak-picked onset - average performance - {}'.format(ave_prf))
            pl = MODEL_DICT[valid_or_test].peak_picked_pl_and_tb_proto['pl']
            tb_proto = MODEL_DICT[valid_or_test].peak_picked_pl_and_tb_proto['tb_proto']
            summary_writer_dict[valid_or_test].add_summary(sess.run(tb_proto, feed_dict={pl:onset_prfs}), global_step)

        def check_all_global_vars_initialized_fn():
            tmp = sess.run(tf.report_uninitialized_variables(tf.global_variables()))
            assert tmp.size == 0

        if MODEL_DICT['config'].train_or_inference.inference is not None:
            save_path = MODEL_DICT['config'].train_or_inference.inference
            tf.train.Saver().restore(sess, save_path)
            check_all_global_vars_initialized_fn()

            sess.run(tf.initializers.variables(tf.local_variables()))
            for model_name in MODEL_DICT['config'].model_names:
                if 'test' in model_name:
                    sess.run(MODEL_DICT[model_name].reinitializable_iter_for_dataset.initializer)
                    logging.info('do inference on {}'.format(model_name))
                    test_or_validate_fn(model_name)

        elif MODEL_DICT['config'].train_or_inference.from_saved is not None:
            save_path = MODEL_DICT['config'].train_or_inference.from_saved
            tf.train.Saver().restore(sess, save_path)
            check_all_global_vars_initialized_fn()

            logging.info('reproduce results ...')
            sess.run(tf.initializers.variables(tf.local_variables()))
            for model_name in MODEL_DICT['config'].model_names:
                if 'training' not in model_name:
                    sess.run(MODEL_DICT[model_name].reinitializable_iter_for_dataset.initializer)

            for model_name in MODEL_DICT['config'].model_names:
                if 'training' not in model_name:
                    logging.info(model_name)
                    test_or_validate_fn(model_name, 0)

        else:  # neither inference or from saved
            logging.info('train from scratch')
            sess.run(tf.initializers.variables(tf.global_variables()))
            check_all_global_vars_initialized_fn()

        if MODEL_DICT['config'].train_or_inference.inference is None:
            check_all_global_vars_initialized_fn()
            if MODEL_DICT['config'].train_or_inference.model_prefix is not None:
                assert 'model_saver' not in MODEL_DICT
                MODEL_DICT['model_saver'] = tf.train.Saver(max_to_keep=200)

            for training_valid_test_epoch_idx in xrange(MODEL_DICT['config'].num_epochs):
                logging.info('\n\ncycle - {}/{}'.format(training_valid_test_epoch_idx + 1, MODEL_DICT['config'].num_epochs))

                sess.run(tf.initializers.variables(tf.local_variables()))

                # to enable prefetch
                for model_name in MODEL_DICT['config'].model_names:
                    if 'training' not in model_name:
                        sess.run(MODEL_DICT[model_name].reinitializable_iter_for_dataset.initializer)

                for training_valid_or_test in MODEL_DICT['config'].model_names:
                    logging.info(training_valid_or_test)

                    if 'training' in training_valid_or_test:
                        ops_per_batch = OP_DICT[training_valid_or_test]['batch']
                        ops_per_epoch = OP_DICT[training_valid_or_test]['epoch']
                        for batch_idx in xrange(MODEL_DICT['config'].batches_per_epoch):
                            sess.run(ops_per_batch)
                            logging.debug('batch - {}/{}'.format(batch_idx + 1, MODEL_DICT['config'].batches_per_epoch))
                        summary_writer_dict[training_valid_or_test].add_summary(
                            sess.run(ops_per_epoch),
                            training_valid_test_epoch_idx + 1
                        )

                        if MODEL_DICT['config'].train_or_inference.model_prefix is not None:
                            save_path = MODEL_DICT['config'].train_or_inference.model_prefix + \
                                        '_' + 'epoch_{}_of_{}'.format(training_valid_test_epoch_idx + 1,
                                                                      MODEL_DICT['config'].num_epochs)
                            save_path = os.path.join('saved_model', save_path)
                            save_path = MODEL_DICT['model_saver'].save(
                                sess=sess,
                                save_path=save_path,
                                global_step=None,
                                write_meta_graph=False
                            )
                            logging.info('model saved to {}'.format(save_path))

                    else:
                        test_or_validate_fn(training_valid_or_test, training_valid_test_epoch_idx + 1)

        msg = 'training end time - {}'.format(datetime.datetime.now())
        logging.info(msg)
        summary_writer_dict[MODEL_DICT['config'].model_names[0]].add_summary(sess.run(aug_info_summary, feed_dict={aug_info_pl: msg}))

        for training_valid_or_test in MODEL_DICT['config'].model_names:
            summary_writer_dict[training_valid_or_test].close()


if __name__ == '__main__':
   main()

















