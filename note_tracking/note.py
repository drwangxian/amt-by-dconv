"""
This script implements our AMT system for note tracking.

How to use this code
1. Train the frame and onset detectors.
2. Create a folder, e.g., maps/note if the training dataset is MAPS, and copy this script to the folder.
   By default, the note level performance for different splits will be saved in folder ./tb_inf. You can view the
   outputs with tensorboard.
3. Create a folder note/saved_model. Then, copy the checkpoint of the frame detector that has the best validation
   performance to this folder, and rename this checkpoint as frame_model. Next, copy the checkpoint of the onset
   detector that has the best validation performance to this folder, and rename this checkpoint as onset_model.
4. Configure the following parameters:
    DEBUG: in {True, False}. If True, will run in a debug mode where only very few recordings will be run. The debug
           mode enables you to quickly check if the script can run correctly.
    GPU_ID: in {0, 1, ..., n - 1} where n is the number of GPUs available.
    TRAINING_DATASET_IS_MAPS: in {True, False}. Set to True if the dataset for training is MAPS, or otherwise to False.
4. Refer to class Config for more options.
"""

from __future__ import print_function
import numpy as np

DEBUG = False
GPU_ID = 1
TRAINING_DATASET_IS_MAPS = True

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import glob
import re
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import magenta.music
import soundfile
from tensorflow.python import pywrap_tensorflow
import collections
import csv
import mido
from mir_eval.transcription import xian_onset_frame_transcription_performance_fn

# the folder where the checkpoints for the frame and onset detectors are stored
ONSET_FRAME_MODEL_DIR = 'saved_model'


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
        assert sr in (16000, 44100)
        spec_stride = int(spec_stride)
        assert spec_stride == 512 if sr == 16000 else 22 * 64
        assert spec_stride & 1 == 0
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        start_frame = (start_sample + spec_stride // 2) // spec_stride
        end_frame = (end_sample + spec_stride // 2 - 1) // spec_stride
        return start_frame, end_frame + 1

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
    def note_seq_to_valued_intervals(note_seq):

        total_num_notes = len(note_seq.notes)
        start_end_times = []
        pitches = []
        for note in note_seq.notes:
            pitches.append(note.pitch)
            start_end_times.append([note.start_time, note.end_time])
        start_end_times = np.asarray(start_end_times, dtype=np.float32).reshape(-1, 2)
        pitches = np.asarray(pitches, dtype=np.uint8).reshape(-1)
        assert start_end_times.shape == (total_num_notes, 2)
        assert len(pitches) == total_num_notes

        return dict(times=start_end_times, pitches=pitches)

    @staticmethod
    def get_note_seq_from_mid_file_fn(mid_file_name):
        note_seq = magenta.music.midi_file_to_note_sequence(mid_file_name)
        note_seq = magenta.music.apply_sustain_control_changes(note_seq)

        return note_seq

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
    def restore_global_vars_fn(sess, model_dir):

        def _var_dict_for_restoring_model_fn(model_dir, frame_or_onset):

            assert frame_or_onset in ('frame', 'onset')
            existing_model_path = os.path.join(model_dir, '{}_model'.format(frame_or_onset))
            reader = pywrap_tensorflow.NewCheckpointReader(existing_model_path)
            existing_model_var_to_shape_map = reader.get_variable_to_shape_map()

            pattern = '{}_detector/'.format(frame_or_onset)
            var_dict = {}
            for var in tf.global_variables():
                name = var.op.name
                if name.startswith(pattern):
                    assert name in existing_model_var_to_shape_map
                    assert var.shape.as_list() == existing_model_var_to_shape_map[name]
                    var_dict[name] = var

            return var_dict

        vars_initialized = []

        for name in ('onset', 'frame'):
            var_dict = _var_dict_for_restoring_model_fn(model_dir, name)
            vars_initialized.extend(var_dict.values())
            model_path = os.path.join(model_dir, '{}_model'.format(name))
            tf.train.Saver(var_dict).restore(sess, model_path)

        assert all(var in vars_initialized for var in tf.global_variables())
        assert len(vars_initialized) == len(tf.global_variables())

    @staticmethod
    def detector_fn(frame_or_onset, inputs, is_training):
        assert frame_or_onset in ('frame', 'onset')
        inputs.set_shape([1, None, 336])
        assert isinstance(is_training, bool)
        assert tf.get_variable_scope().name == ''

        with tf.variable_scope('{}_detector'.format(frame_or_onset), reuse=tf.AUTO_REUSE):
            with tf.variable_scope('cnn_layers'):
                cnn_features = MiscFns.acoustic_model_fn(
                    spec_batch=inputs,
                    is_training=is_training,
                    use_feature=True,
                    trainable=False
                )
                cnn_features.set_shape([1, None, 88, 64])

            with tf.variable_scope('rnn_layer'):
                rnn_features = MiscFns.rnn_layer_fn(inputs=cnn_features, trainable=False)
                rnn_features.set_shape([1, None, 88, 64])

            with tf.variable_scope('output_layer'):
                logits = slim.fully_connected(
                    scope='FC',
                    inputs=rnn_features,
                    num_outputs=1,
                    activation_fn=None,
                    trainable=False
                )
                logits = tf.squeeze(logits, axis=-1)
                logits.set_shape([1, None, 88])

        return logits

    @staticmethod
    def frame_label_detector_fn(inputs, is_training):
        return MiscFns.detector_fn(frame_or_onset='frame', inputs=inputs, is_training=is_training)

    @staticmethod
    def onset_detector_fn(inputs, is_training):
        return MiscFns.detector_fn(frame_or_onset='onset', inputs=inputs, is_training=is_training)

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
    def maps_sg_and_note_seq_fn(wav_file):

        wav_info = soundfile.info(wav_file)
        assert wav_info.samplerate == 44100
        num_frames = MiscFns.num_samples_to_num_frames_fn(wav_info.frames)

        rec_name = os.path.basename(wav_file)[:-4]
        vqt_file = os.path.join(os.environ['maps_vqt'], rec_name + '.vqt')
        _rec_name, vqt = MiscFns.load_np_array_from_file_fn(vqt_file)
        assert _rec_name == rec_name
        assert vqt.dtype == np.float32 and vqt.shape == (num_frames, 336)

        mid_file = wav_file[:-3] + 'mid'
        num_frames_from_midi = mido.MidiFile(mid_file).length
        num_frames_from_midi = int(np.ceil(num_frames_from_midi * wav_info.samplerate))
        num_frames_from_midi = MiscFns.num_samples_to_num_frames_fn(num_frames_from_midi)
        num_frames_from_midi += 2
        num_frames = min(num_frames, num_frames_from_midi)

        vqt = vqt[:num_frames]
        vqt = np.require(vqt, dtype=np.float32, requirements=['O', 'C'])
        vqt.flags['WRITEABLE'] = False

        note_seq = MiscFns.get_note_seq_from_mid_file_fn(mid_file_name=mid_file)
        times_and_pitches = MiscFns.note_seq_to_valued_intervals(note_seq)
        note_intervals = times_and_pitches['times']
        note_intervals.flags['WRITEABLE'] = False
        note_pitches = times_and_pitches['pitches']
        note_pitches.flags['WRITEABLE'] = False
        assert len(note_intervals) == len(note_pitches)

        return dict(sg=vqt, intervals=note_intervals, pitches=note_pitches)

    @staticmethod
    def maestro_sg_and_note_seq_fn(year, rec_name):

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

        num_frames = _get_num_frames_fn(year, rec_name)
        vqt_file = os.path.join(os.environ['maestro_vqt'], year, rec_name + '.vqt')
        _rec_name, vqt = MiscFns.load_np_array_from_file_fn(vqt_file)
        assert _rec_name == rec_name
        assert vqt.dtype == np.float32 and vqt.shape == (num_frames, 336)

        sr = 44100
        mid_file = os.path.join(os.environ['maestro'], year, rec_name + '.midi')
        num_frames_from_midi = mido.MidiFile(mid_file).length
        num_frames_from_midi = int(np.ceil(num_frames_from_midi * sr))
        num_frames_from_midi = MiscFns.num_samples_to_num_frames_fn(num_frames_from_midi)
        num_frames_from_midi += 2
        num_frames = min(num_frames, num_frames_from_midi)

        vqt = vqt[:num_frames]
        vqt = np.require(vqt, dtype=np.float32, requirements=['O', 'C'])
        vqt.flags['WRITEABLE'] = False

        note_seq = MiscFns.get_note_seq_from_mid_file_fn(mid_file_name=mid_file)
        times_and_pitches = MiscFns.note_seq_to_valued_intervals(note_seq)
        note_intervals = times_and_pitches['times']
        note_intervals.flags['WRITEABLE'] = False
        note_pitches = times_and_pitches['pitches']
        note_pitches.flags['WRITEABLE'] = False
        assert len(note_intervals) == len(note_pitches)

        return dict(sg=vqt, intervals=note_intervals, pitches=note_pitches)

    @staticmethod
    def tb_proto_for_note_level_performance_fn(header, tf_name_scope, tb_scope, title, names, dtype):
        num_examples = len(names)
        num_fields = len(header)
        with tf.name_scope(tf_name_scope):
            tf_pl = tf.placeholder(dtype=dtype, shape=[num_examples, num_fields])
            table_proto = MiscFns.array_to_table_tf_fn(
                tf_array=tf_pl,
                header=header,
                scope=tb_scope,
                title=title,
                names=names
            )
            ind_fields = tf.unstack(tf_pl[-1])
            scalar_portos = []
            for name, data in zip(header, ind_fields):
                scalar_portos.append(tf.summary.scalar(name, data))
            scalar_portos.append(table_proto)
        pl_and_tb_proto = dict(
            pl=tf_pl,
            tb_proto=tf.summary.merge(scalar_portos)
        )

        return pl_and_tb_proto


# all configurations go here
class Config(object):

    def __init__(self):
        self.debug_mode = DEBUG
        self.gpu_id = GPU_ID
        self.snippet_len = 1200
        self.tb_dir = 'tb_inf'
        self.model_dir = ONSET_FRAME_MODEL_DIR

        # check if folder for storing tensorboard data exists beforehand
        if not self.debug_mode:
            # check if tb_dir exists
            assert self.tb_dir is not None
            tmp_dirs = glob.glob('./*/')
            tmp_dirs = [s[2:-1] for s in tmp_dirs]
            assert self.tb_dir not in tmp_dirs

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.gpu_config = gpu_config

        # get splits for training, validation and test
        split_maestro = MiscFns.split_train_valid_and_test_files_fn('maestro')
        split_maps = MiscFns.split_train_valid_and_test_files_fn('maps')
        tvt_split_dict = dict(
            validation_maestro=split_maestro['validation'],
            validation_maps=split_maps['validation'],
            test_maestro=split_maestro['test'],
            test_maps=split_maps['test']
        )
        self.training_dataset_is_maps = TRAINING_DATASET_IS_MAPS
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
                _sel = np.random.choice(_num, 2, replace=False)
                self.tvt_split_dict[tvt] = [self.tvt_split_dict[tvt][ii] for ii in _sel]

        
# define neural network models
class Model(object):
    def __init__(self, config, name):
        assert name in config.model_names
        self.name = name
        self.is_training = True if 'training' in self.name else False
        self.config = config
        self.batch = self._gen_batch_fn()

        with tf.name_scope(self.name):
            self.logits = {}
            for task in ('onset', 'frame'):
                with tf.name_scope(task):
                    self.logits[task] = self._nn_model_fn(task)

            self.rec_names = tuple(self._get_rec_names_fn())
            names = list(self.rec_names) + ['average']
            self.pl_and_tb_proto = {}
            for w_or_wo in ('with', 'without'):
                description = 'note_level_performance_{}_offset'.format(w_or_wo)
                pl_and_tb_proto = MiscFns.tb_proto_for_note_level_performance_fn(
                    header=['p', 'r', 'f', 'o'],
                    tf_name_scope=description,
                    tb_scope=description,
                    title=description.replace('_', ' '),
                    dtype=tf.float32,
                    names=names
                )
                self.pl_and_tb_proto[w_or_wo] = pl_and_tb_proto

            description = 'frame_level_performance'
            pl_and_tb_proto = MiscFns.tb_proto_for_note_level_performance_fn(
                header=['p', 'r', 'f'],
                tf_name_scope=description,
                tb_scope=description,
                title=description.replace('_', ' '),
                dtype=tf.float32,
                names=names
            )
            self.pl_and_tb_proto['frame'] = pl_and_tb_proto

    def _get_rec_names_fn(self):

        tvt, dataset_name = self.name.split('_')
        assert tvt in ('training', 'validation', 'test')
        assert dataset_name in ('maestro', 'maps')
        dataset_is_maps = dataset_name == 'maps'
        if dataset_is_maps:
            rec_names = []
            for wav_file in self.config.tvt_split_dict[self.name]:
                rec_names.append(os.path.basename(wav_file)[:-4])
        else:
            rec_names = []
            for _, rec_name in self.config.tvt_split_dict[self.name]:
                rec_names.append(rec_name)

        return rec_names

    def _nn_model_fn(self, task):
        assert task in ('frame', 'onset')
        inputs = self.batch['spectrogram']
        inputs = tf.ensure_shape(inputs, [1, None, 336])

        _nn_fn = dict(
            onset=MiscFns.onset_detector_fn,
            frame=MiscFns.frame_label_detector_fn
        )
        _nn_fn = _nn_fn[task]
        outputs = _nn_fn(inputs=inputs, is_training=self.is_training)
        outputs = tf.stop_gradient(outputs)
        outputs = tf.ensure_shape(outputs, [1, None, 88])

        return outputs

    def _dataset_iter_fn(self):

        logging.debug('{} - enter dataset generator'.format(self.name))

        tvt, dataset_name = self.name.split('_')
        assert tvt in ('training', 'validation', 'test')
        assert dataset_name in ('maestro', 'maps')

        dataset_is_maps = dataset_name == 'maps'
        file_names = self.config.tvt_split_dict[self.name]
        num_recs = len(file_names)
        dummy_intervals = np.asarray([[-2, -1]], dtype=np.float32)
        dummy_pitches = np.asarray([0], dtype=np.uint8)

        if dataset_is_maps:
            for rec_idx, wav_file in enumerate(file_names):
                rec_name = os.path.basename(wav_file)[:-4]
                logging.debug('{}/{} - {}'.format(rec_idx + 1, num_recs, rec_name))
                sg_intervals_pitches = MiscFns.maps_sg_and_note_seq_fn(wav_file=wav_file)
                sg = sg_intervals_pitches['sg']
                assert not sg.flags['WRITEABLE']
                assert sg.dtype == np.float32 and sg.shape[1:] == (336,)
                intervals = sg_intervals_pitches['intervals']
                pitches = sg_intervals_pitches['pitches']

                split_list = MiscFns.gen_split_list_fn(len(sg), self.config.snippet_len)
                num_snippets = len(split_list)
                total_num_frames = len(sg)

                for snippet_idx, (s, e) in enumerate(split_list):
                    yield dict(
                        rec_idx=rec_idx,
                        num_snippets=num_snippets,
                        snippet_idx=snippet_idx,
                        total_num_frames=total_num_frames,
                        num_frames=e - s,
                        spectrogram=sg[s:e],
                        intervals=dummy_intervals if snippet_idx < num_snippets - 1 else intervals,
                        pitches=dummy_pitches if snippet_idx < num_snippets - 1 else pitches
                    )
        else:  # dataset is maestro
            for rec_idx, (year, rec_name) in enumerate(file_names):
                logging.debug('{}/{} - {}'.format(rec_idx + 1, num_recs, rec_name))
                sg_intervals_pitches = MiscFns.maestro_sg_and_note_seq_fn(year, rec_name)
                sg = sg_intervals_pitches['sg']
                assert not sg.flags['WRITEABLE']
                assert sg.dtype == np.float32 and sg.shape[1:] == (336,)

                intervals = sg_intervals_pitches['intervals']
                pitches = sg_intervals_pitches['pitches']

                split_list = MiscFns.gen_split_list_fn(len(sg), self.config.snippet_len)
                num_snippets = len(split_list)
                total_num_frames = len(sg)

                for snippet_idx, (s, e) in enumerate(split_list):
                    yield dict(
                        rec_idx=rec_idx,
                        num_snippets=num_snippets,
                        snippet_idx=snippet_idx,
                        total_num_frames=total_num_frames,
                        num_frames=e - s,
                        spectrogram=sg[s:e],
                        intervals=dummy_intervals if snippet_idx < num_snippets - 1 else intervals,
                        pitches=dummy_pitches if snippet_idx < num_snippets - 1 else pitches
                    )

    def _gen_batch_fn(self):
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_generator(
                generator=self._dataset_iter_fn,
                output_types=dict(
                        rec_idx=tf.int32,
                        num_snippets=tf.int32,
                        snippet_idx=tf.int32,
                        total_num_frames=tf.int32,
                        num_frames=tf.int32,
                        spectrogram=tf.float32,
                        intervals=tf.float32,
                        pitches=tf.uint8
                    ),
                output_shapes=dict(
                        rec_idx=[],
                        num_snippets=[],
                        snippet_idx=[],
                        total_num_frames=[],
                        num_frames=[],
                        spectrogram=[None, 336],
                        intervals=[None, 2],
                        pitches=[None]
                    )
            )
            dataset = dataset.batch(1)
            dataset = dataset.prefetch(20)
            it = dataset.make_one_shot_iterator()
            element = it.get_next()

        return element


def main():
    MODEL_DICT = {}
    MODEL_DICT['config'] = Config()
    for model_name in MODEL_DICT['config'].model_names:
        MODEL_DICT[model_name] = Model(config=MODEL_DICT['config'], name=model_name)

    aug_info_pl = tf.placeholder(dtype=tf.string, name='aug_info_pl')
    aug_info_summary = tf.summary.text('aug_info_summary', aug_info_pl)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(MODEL_DICT['config'].gpu_id)
    with tf.Session(config=MODEL_DICT['config'].gpu_config) as sess:
        # summary writer
        summary_writer_dict = {}
        for model_name in MODEL_DICT['config'].model_names:
            summary_writer_dict[model_name] = tf.summary.FileWriter(
                os.path.join(MODEL_DICT['config'].tb_dir, model_name)
            )

        aug_info = []

        aug_info.append('note tracking performance')
        training_dataset = MODEL_DICT['config'].training_dataset_is_maps
        training_dataset = 'maps' if training_dataset else 'maestro'
        aug_info.append('trained on - {}'.format(training_dataset))

        aug_info.append('tb dir - {}'.format(MODEL_DICT['config'].tb_dir))
        aug_info.append('debug mode - {}'.format(MODEL_DICT['config'].debug_mode))

        aug_info = '\n\n'.join(aug_info)
        logging.info(aug_info)
        summary_writer_dict[MODEL_DICT['config'].model_names[0]].add_summary(sess.run(aug_info_summary, feed_dict={aug_info_pl: aug_info}))

        OP_DICT = {}
        for model_name in MODEL_DICT['config'].model_names:
            m = MODEL_DICT[model_name]
            batch_op_dict = dict()
            batch_op_dict['logits'] = m.logits
            for k in ('rec_idx', 'num_snippets', 'snippet_idx', 'total_num_frames', 'num_frames', 'intervals', 'pitches'):
                batch_op_dict[k] = m.batch[k]
            tmp = dict(
                batch=batch_op_dict,
                epoch=m.pl_and_tb_proto
            )
            OP_DICT[model_name] = tmp

        def inference_fn(model_name):
            assert model_name in MODEL_DICT['config'].model_names

            ops_per_batch = OP_DICT[model_name]['batch']
            ops_per_epoch = OP_DICT[model_name]['epoch']

            num_recs = len(MODEL_DICT[model_name].rec_names)

            prfos = dict()
            prfos['with'] = []
            prfos['without'] = []
            prfos['frame'] = []
            for rec_idx in xrange(num_recs):
                rec_logits = {}
                for n in ('onset', 'frame'):
                    rec_logits[n] = []

                while True:
                    tmp = sess.run(ops_per_batch)
                    num_snippets = tmp['num_snippets'][0]
                    snippet_idx = tmp['snippet_idx'][0]
                    _rec_idx = tmp['rec_idx'][0]
                    assert _rec_idx == rec_idx
                    logits = tmp['logits']
                    num_frames = tmp['num_frames'][0]
                    for n, v in logits.iteritems():
                        assert v.shape == (1, num_frames, 88)
                        rec_logits[n].append(np.squeeze(v, axis=0))
                    if snippet_idx == num_snippets - 1:
                        break
                total_num_frames = tmp['total_num_frames'][0]
                for n in ('onset', 'frame'):
                    rec_logits[n] = np.concatenate(rec_logits[n], axis=0)
                    assert rec_logits[n].shape == (total_num_frames, 88)
                prfo_dict = xian_onset_frame_transcription_performance_fn(
                    logits_dict=rec_logits,
                    ref_intervals=np.squeeze(tmp['intervals'], axis=0),
                    ref_notes=np.squeeze(tmp['pitches'], axis=0),
                    sr=44100,
                    hop_size=22 * 64
                )
                rec_name = MODEL_DICT[model_name].rec_names[rec_idx]
                logging.info('{}/{} - {}:'.format(rec_idx + 1, num_recs, rec_name))

                for n in ('without', 'with'):
                    v = prfo_dict[n]
                    prfos[n].append(v)
                    logging.info('  note level - {} offset - {}'.format(n, v))

                n = 'frame'
                v = prfo_dict['frame']
                prfos[n].append(v)
                logging.info('  frame level - {}'.format(v))

            for w_or_wo_offset in ('without', 'with'):
                pl = ops_per_epoch[w_or_wo_offset]['pl']
                tb_proto = ops_per_epoch[w_or_wo_offset]['tb_proto']
                v = prfos[w_or_wo_offset]
                v = np.asarray(v)
                av = np.mean(v, axis=0)
                logging.info('note level performance - {} offset - average - {}'.format(w_or_wo_offset, av))
                v = np.concatenate([v, av[None, :]], axis=0)
                s = sess.run(tb_proto, feed_dict={pl: v})
                summary_writer_dict[model_name].add_summary(s)

            v = prfos['frame']
            v = np.asarray(v)
            av = np.mean(v, axis=0)
            logging.info('frame level performance - mean - {}'.format(av))
            v = np.concatenate([v, av[None, :]], axis=0)
            pl = ops_per_epoch['frame']['pl']
            tb_proto = ops_per_epoch['frame']['tb_proto']
            s = sess.run(tb_proto, feed_dict={pl: v})
            summary_writer_dict[model_name].add_summary(s)

        def check_all_global_vars_initialized_fn():
            tmp = sess.run(tf.report_uninitialized_variables(tf.global_variables()))
            assert not tmp

        assert not tf.trainable_variables()
        assert not tf.local_variables()
        MiscFns.restore_global_vars_fn(sess=sess, model_dir=MODEL_DICT['config'].model_dir)
        check_all_global_vars_initialized_fn()

        logging.info('do inference ...')
        for model_name in MODEL_DICT['config'].model_names:
            logging.info(model_name)
            inference_fn(model_name)

        for model_name in MODEL_DICT['config'].model_names:
            summary_writer_dict[model_name].close()


if __name__ == '__main__':
   main()


















