# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import argparse
import logging
import random
import codecs
import math
import yaml
import copy
import json

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchaudio.compliance.kaldi as kaldi
import torchaudio
from PIL import Image
from PIL.Image import BICUBIC

from wenet.utils.common import IGNORE_ID
from wenet.dataset.wav_distortion import distort_wav_conf
import wenet.dataset.kaldi_io as kaldi_io


def _splice(feats, left_context, right_context):
    """ Splice feature

    Args:
        feats: input feats
        left_context: left context for splice
        right_context: right context for splice

    Returns:
        Spliced feature
    """
    if left_context == 0 and right_context == 0:
        return feats
    assert (len(feats.shape) == 2)
    num_rows = feats.shape[0]
    first_frame = feats[0]
    last_frame = feats[-1]
    padding_feats = feats
    if left_context > 0:
        left_padding = np.vstack([first_frame for i in range(left_context)])
        padding_feats = np.vstack((left_padding, padding_feats))
    if right_context > 0:
        right_padding = np.vstack([last_frame for i in range(right_context)])
        padding_feats = np.vstack((padding_feats, right_padding))
    outputs = []
    for i in range(num_rows):
        splice_feats = np.hstack([
            padding_feats[i]
            for i in range(i, i + 1 + left_context + right_context)
        ])
        outputs.append(splice_feats)
    return np.vstack(outputs)

def _load_json_cmvn(json_cmvn_file):
    """ Load the json format cmvn stats file and calculate cmvn

    Args:
        json_cmvn_file: cmvn stats file in json format

    Returns:
        a numpy array of [means, vars]
    """
    with open(json_cmvn_file) as f:
        cmvn_stats = json.load(f)

    means = cmvn_stats['mean_stat']
    variance = cmvn_stats['var_stat']
    count = cmvn_stats['frame_num']
    for i in range(len(means)):
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])
    cmvn = np.array([means, variance])
    return cmvn

def _load_kaldi_cmvn(kaldi_cmvn_file):
    """ Load the kaldi format cmvn stats file and calculate cmvn

    Args:
        kaldi_cmvn_file:  kaldi text style global cmvn file, which
           is generated by:
           compute-cmvn-stats --binary=false scp:feats.scp global_cmvn

    Returns:
        a numpy array of [means, vars]
    """
    means = []
    variance = []
    with open(kaldi_cmvn_file, 'r') as fid:
        # kaldi binary file start with '\0B'
        if fid.read(2) == '\0B':
            logging.error('kaldi cmvn binary file is not supported, please '
                          'recompute it by: compute-cmvn-stats --binary=false '
                          ' scp:feats.scp global_cmvn')
            sys.exit(1)
        fid.seek(0)
        arr = fid.read().split()
        assert (arr[0] == '[')
        assert (arr[-2] == '0')
        assert (arr[-1] == ']')
        feat_dim = int((len(arr) - 2 - 2) / 2)
        for i in range(1, feat_dim + 1):
            means.append(float(arr[i]))
        count = float(arr[feat_dim + 1])
        for i in range(feat_dim + 2, 2 * feat_dim + 2):
            variance.append(float(arr[i]))

    for i in range(len(means)):
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])
    cmvn = np.array([means, variance])
    return cmvn


def _spec_augmentation(x,
                       warp_for_time=False,
                       num_t_mask=2,
                       num_f_mask=2,
                       max_t=50,
                       max_f=10,
                       max_w=80):
    """ Deep copy x and do spec augmentation then return it

    Args:
        x: input feature, T * F 2D
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
        max_w: max width of time warp

    Returns:
        augmented feature
    """
    y = np.copy(x)
    max_frames = y.shape[0]
    max_freq = y.shape[1]

    # time warp
    if warp_for_time and max_frames > max_w * 2:
        center = random.randrange(max_w, max_frames - max_w)
        warped = random.randrange(center - max_w, center + max_w) + 1

        left = Image.fromarray(x[:center]).resize((max_freq, warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((max_freq,
                                                   max_frames - warped),
                                                   BICUBIC)
        y = np.concatenate((left, right), 0)
    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for i in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    return y

def _waveform_distortion(waveform, distortion_methods_conf):
    """ Apply distortion on waveform

    This distortion will not change the length of the waveform.

    Args:
        waveform: numpy float tensor, (length,)
        distortion_methods_conf: a list of config for ditortion method.
            a method will be randomly selected by 'method_rate' and
            apply on the waveform.

    Returns:
        distorted waveform.
    """
    r = random.uniform(0, 1)
    acc = 0.0
    for distortion_method in distortion_methods_conf:
        method_rate = distortion_method['method_rate']
        acc += method_rate
        if r < acc:
            distortion_type = distortion_method['name']
            distortion_conf = distortion_method['params']
            point_rate = distortion_method['point_rate']
            return distort_wav_conf(waveform, distortion_type,
                                    distortion_conf , point_rate)
    return waveform

# add speed perturb when loading wav
# return augmented, sr
def _load_wav_with_speed(wav_file, speed):
    """ Load the wave from file and apply speed perpturbation

    Args:
        wav_file: input feature, T * F 2D

    Returns:
        augmented feature
    """
    if speed == 1.0:
        return torchaudio.load_wav(wav_file)
    else:
        si, _ = torchaudio.info(wav_file)
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.append_effect_to_chain('speed', speed)
        E.append_effect_to_chain("rate", si.rate)
        E.set_input_file(wav_file)
        return E.sox_build_flow_effects()



def _extract_feature(batch, speed_perturb, wav_distortion_conf,
                     feature_extraction_conf):
    """ Extract acoustic fbank feature from origin waveform.

    Speed perturbation and wave amplitude distortion is optional.

    Args:
        batch: a list of tuple (wav id , wave path).
        speed_perturb: bool, whether or not to use speed pertubation.
        wav_distortion_conf: a dict , the config of wave amplitude distortion.
        feature_extraction_conf:a dict , the config of fbank extraction.

    Returns:
        (keys, feats, labels)
    """
    keys = []
    feats = []
    lengths = []
    wav_dither = wav_distortion_conf['wav_dither'] / 32768.0
    wav_distortion_rate = wav_distortion_conf['wav_distortion_rate']
    distortion_methods_conf = wav_distortion_conf['distortion_methods']
    if speed_perturb:
        speeds = [1.0, 1.1, 0.9]
        weights = [1, 1, 1]
        speed = random.choices(speeds, weights, k=1)[0]
        # speed = random.choice(speeds)
    for i, x in enumerate(batch):
        try:
            if speed_perturb:
                waveform, sample_rate = _load_wav_with_speed(x[1], speed)
            else:
                waveform, sample_rate = torchaudio.load_wav(x[1])
            if wav_distortion_rate > 0.0:
                r = random.uniform(0, 1)
                if r < wav_distortion_rate:
                    waveform = waveform.detach().numpy()
                    waveform = _waveform_distortion(waveform,
                                                    distortion_methods_conf)
                    waveform = torch.from_numpy(waveform)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=feature_extraction_conf['mel_bins'],
                frame_length=feature_extraction_conf['frame_length'],
                frame_shift=feature_extraction_conf['frame_shift'],
                dither=wav_dither,
                energy_floor=0.0
            )
            mat = mat.detach().numpy()
            feats.append(mat)
            keys.append(x[0])
            lengths.append(mat.shape[0])
        except (Exception) as e:
            print(e)
            logging.warn('read utterance {} error'.format(x[0]))
            pass
    # Sort it because sorting is required in pack/pad operation
    order = np.argsort(lengths)[::-1]
    sorted_keys = [keys[i] for i in order]
    sorted_feats = [feats[i] for i in order]
    labels = [x[2].split() for x in batch]
    labels = [np.fromiter(map(int, x), dtype=np.int32) for x in labels]
    sorted_labels = [labels[i] for i in order]
    return sorted_keys, sorted_feats, sorted_labels


def _load_feature(batch):
    """ Load acoustic feature from files.

    The features have been prepared in previous step, usualy by Kaldi.

    Args:
        batch: a list of tuple (wav id , feature ark path).

    Returns:
        (keys, feats, labels)
    """
    keys = []
    feats = []
    lengths = []
    for i, x in enumerate(batch):
        try:
            mat = kaldi_io.read_mat(x[1])
            feats.append(mat)
            keys.append(x[0])
            lengths.append(mat.shape[0])
        except (Exception):
            # logging.warn('read utterance {} error'.format(x[0]))
            pass
    # Sort it because sorting is required in pack/pad operation
    order = np.argsort(lengths)[::-1]
    sorted_keys = [keys[i] for i in order]
    sorted_feats = [feats[i] for i in order]
    labels = [x[2].split() for x in batch]
    labels = [np.fromiter(map(int, x), dtype=np.int32) for x in labels]
    sorted_labels = [labels[i] for i in order]
    return sorted_keys, sorted_feats, sorted_labels

class CollateFunc(object):
    """ Collate function for AudioDataset
    """
    def __init__(self,
                 cmvn=None,
                 subsampling_factor=1,
                 left_context=0,
                 right_context=0,
                 feature_dither=0.0,
                 speed_perturb=False,
                 spec_aug=False,
                 spec_aug_conf=None,
                 raw_wav=True,
                 feature_extraction_conf=None,
                 wav_distortion_conf=None,
                 ):
        """
        Args:
            subsampling_factor: subsampling_factor for feature
            left_context: left context for splice feature
            right_context: right_context for splice feature
            raw_wav:
                    True if input is raw wav and feature extraction is needed.
                    False if input is extracted feature
        """
        self.cmvn = None
        if cmvn is not None:
            if raw_wav:
                self.cmvn = _load_json_cmvn(cmvn)
            else:
                self.cmvn = _load_kaldi_cmvn(cmvn)

        self.wav_distortion_conf = wav_distortion_conf
        self.feature_extraction_conf = feature_extraction_conf
        self.subsampling_factor = subsampling_factor
        self.left_context = left_context
        self.right_context = right_context
        self.spec_aug = spec_aug
        self.feature_dither = feature_dither
        self.speed_perturb = speed_perturb
        self.raw_wav = raw_wav
        self.spec_aug_conf = spec_aug_conf

    def __call__(self, batch):
        assert (len(batch) == 1)
        if self.raw_wav:
            keys, xs, ys = _extract_feature(batch[0],
                                            self.speed_perturb,
                                            self.wav_distortion_conf,
                                            self.feature_extraction_conf)

        else:
            keys, xs, ys = _load_feature(batch[0])

        train_flag = True
        if ys is None:
            train_flag = False

        # optional cmvn
        if self.cmvn is not None:
            # if self.norm_mean:
            xs = [x - self.cmvn[0] for x in xs]
            # if self.norm_var:
            xs = [x * self.cmvn[1] for x in xs]

        # optional feature dither d ~ (-a, a) on fbank feature
        # a ~ (0, 0.5)
        if self.feature_dither != 0.0:
            a = random.uniform(0, self.feature_dither)
            xs = [x + (np.random.random_sample(x.shape) - 0.5) * a for x in xs]

        # optinoal spec augmentation
        if self.spec_aug:
            xs = [_spec_augmentation(x, **self.spec_aug_conf) for x in xs]

        # optional splice
        if self.left_context != 0 or self.right_context != 0:
            xs = [
                _splice(x, self.left_context, self.right_context) for x in xs
            ]

        # optional subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor] for x in xs]

        # padding
        xs_lengths = torch.from_numpy(
            np.array([x.shape[0] for x in xs], dtype=np.int32))

        # pad_sequence will FAIL in case xs is empty
        if len(xs) > 0:
            xs_pad = pad_sequence([torch.from_numpy(x).float() for x in xs],
                                  True, 0)
        else:
            xs_pad = torch.Tensor(xs)
        if train_flag:
            ys_lengths = torch.from_numpy(
                np.array([y.shape[0] for y in ys], dtype=np.int32))
            if len(ys) > 0:
                ys_pad = pad_sequence([torch.from_numpy(y).int() for y in ys],
                                      True, IGNORE_ID)
            else:
                ys_pad = torch.Tensor(ys)
        else:
            ys_pad = None
            ys_lengths = None
        return keys, xs_pad, ys_pad, xs_lengths, ys_lengths


class AudioDataset(Dataset):
    def __init__(self,
                 data_file,
                 max_length=10240,
                 min_length=0,
                 batch_type='static',
                 batch_size=1,
                 max_frames_in_batch=0,
                 sort=True,
                 raw_wav=True):
        """Dataset for loading audio data.

        Attributes::
            data_file: input data file
                Plain text data file, each line contains following 7 fields,
                which is split by '\t':
                    utt:utt1
                    feat:tmp/data/file1.wav or feat:tmp/data/fbank.ark:30
                    feat_shape: 4.95(in seconds) or feat_shape:495,80(495 is in frames)
                    text:i love you
                    token: i <space> l o v e <space> y o u
                    tokenid: int id of this token
                    token_shape: M,N    # M is the number of token, N is vocab size
            max_length: drop utterance which is greater than max_length(ms)
            min_length: drop utterance which is less than min_length(ms)
            batch_type: static or dynamic, see max_frames_in_batch(dynamic)
            batch_size: number of utterances in a batch,
               it's for static batch size.
            max_frames_in_batch: max feature frames in a batch,
               when batch_type is dynamic, it's for dynamic batch size.
               Then batch_size is ignored, we will keep filling the
               batch until the total frames in batch up to max_frames_in_batch.
            sort: whether to sort all data, so the utterance with the same
               length could be filled in a same batch.
            raw_wav: use raw wave or extracted featute.
                if raw wave is used, dynamic waveform-level augmentation could be used
                and the feature is extracted by torchaudio.
                if extracted featute(e.g. by kaldi) is used, only feature-level
                augmentation such as specaug could be used.
        """
        assert batch_type in ['static', 'dynamic']
        data = []

        # Open in utf8 mode since meet encoding problem
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split('\t')
                if len(arr) != 7:
                    continue
                key = arr[0].split(':')[1]
                tokenid = arr[5].split(':')[1]
                output_dim = int(arr[6].split(':')[1].split(',')[1])
                if raw_wav:
                    wav_path = ':'.join(arr[1].split(':')[1:])
                    duration = int(float(arr[2].split(':')[1]) * 1000)
                    data.append((key, wav_path, duration, tokenid))
                else:
                    feat_ark = ':'.join(arr[1].split(':')[1:])
                    feat_info = arr[2].split(':')[1].split(',')
                    feat_dim = int(feat_info[1].strip())
                    num_frames = int(feat_info[0].strip())
                    data.append((key, feat_ark, num_frames, tokenid))
                    self.input_dim = feat_dim
                self.output_dim = output_dim
        if sort:
            data = sorted(data, key=lambda x: x[2])
        valid_data = []
        for i in range(len(data)):
            length = data[i][2]
            if length > max_length or length < min_length:
                # logging.warn('ignore utterance {} feature {}'.format(
                #     data[i][0], length))
                pass
            else:
                valid_data.append(data[i])
        data = valid_data
        self.minibatch = []
        num_data = len(data)
        # Dynamic batch size
        if batch_type == 'dynamic':
            assert (max_frames_in_batch > 0)
            self.minibatch.append([])
            num_frames_in_batch = 0
            for i in range(num_data):
                length = data[i][2]
                num_frames_in_batch += length
                if num_frames_in_batch > max_frames_in_batch:
                    self.minibatch.append([])
                    num_frames_in_batch = length
                self.minibatch[-1].append((data[i][0], data[i][1], data[i][3]))
        # Static batch size
        else:
            cur = 0
            while cur < num_data:
                end = min(cur + batch_size, num_data)
                item = []
                for i in range(cur, end):
                    item.append((data[i][0], data[i][1], data[i][3]))
                self.minibatch.append(item)
                cur = end

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, idx):
        return self.minibatch[idx]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='config file')
    parser.add_argument('config_file', help='config file')
    parser.add_argument('data_file', help='input data file')
    args = parser.parse_args()

    with open(args.config_file, 'r') as fin:
        configs = yaml.load(fin)

    # Init dataset and data loader
    collate_conf = copy.copy(configs['collate_conf'])
    if args.type == 'raw_wav':
        raw_wav = True
    else:
        raw_wav = False
    collate_func = CollateFunc(**collate_conf, raw_wav=raw_wav)
    dataset_conf = configs.get('dataset_conf', {})
    dataset = AudioDataset(args.data_file, **dataset_conf, raw_wav=raw_wav)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=True,
                             sampler=None,
                             num_workers=0,
                             collate_fn=collate_func)

    for i, batch in enumerate(data_loader):
        print(i)
        # print(batch[1].shape)