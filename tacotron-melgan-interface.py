# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         tacotron-melgan-interface.py
# Description:
# Author:       zy07898
# Date:         2021/3/4 13:42
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
import os, re, argparse,glob, torch,librosa #, librosa.display
from hparams import hparams1
from models import create_model
from util.text import text_to_sequence, sequence_to_text
from util import plot
# import matplotlib.pyplot as plt
from tqdm import tqdm
from gan import Generator
from pathlib import Path
from scipy.io.wavfile import write
from util.audio import *

class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    # tf.reset_default_graph()
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams1)
      self.model.initialize(inputs, input_lengths)
      self.alignments = self.model.alignments[0]
      self.inputs = self.model.inputs[0]
      self.mel_outputs = self.model.mel_outputs[0]

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, args, text, base_path, idx):
    seq = text_to_sequence(text)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    # print('----------1')
    input_seq, alignment, mel = self.session.run([self.inputs, self.alignments, self.mel_outputs], feed_dict=feed_dict)
    input_seq = sequence_to_text(input_seq)
    mel_spectrogram = torch.from_numpy(mel.T)

    # wav = inv_mel_spectrogram(mel.T, hparams)
    # audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}-mel.wav'.format(basenames[i])), sr=hparams.sample_rate)

    plot.plot_alignment(alignment, '%s-%d-align.png' % (base_path, idx), input_seq)
    save_path = os.path.join(args.mel_save_dir, 'mel-%d.mel' % (idx + 1))
    torch.save(mel_spectrogram, save_path)
    return mel_spectrogram


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args,sentences):
  synth = Synthesizer()
  # 改，取模型路径中checkpoint中记载的模型
  taco_checkpoint = tf.train.get_checkpoint_state(args.checkpoint).model_checkpoint_path
  synth.load(taco_checkpoint)

  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    # jamo = ''.join(list(hangul_to_jamo(text)))
    synth.synthesize(args, text, base_path, i)

def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(22050 * min_silence_sec)
  hop_length = window_length // 4
  threshold = np.power(10.0, threshold_db * 0.05) * 32768
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x : x + window_length]) < threshold:
      return x + hop_length
  return len(wav)


def init_tacotron(args):
    synth = Synthesizer()
    # 改，取模型路径中checkpoint中记载的模型
    taco_checkpoint = tf.train.get_checkpoint_state(args.checkpoint).model_checkpoint_path
    synth.load(taco_checkpoint)
    base_path = get_output_base_path(args.checkpoint)

    return synth,base_path

def init_melgan(args):
    ckpt = torch.load(args.mel_load_dir, map_location=torch.device('cpu'))
    # ckpt = torch.load(args.mel_load_dir)

    vocoder = Generator(80)
    vocoder = vocoder

    vocoder.load_state_dict(ckpt['G'])

    return vocoder

# 'tou2 shi2 wen4 lu4 jie2 guo3 bei4 za2 si3 le'
# 'ya2 dian3 cheng1 wang2 de5 han2 guo2 xiao3 zi5 liu3 cheng2 min3 you3 wang4 chuang4 zao4 li4 shi3 ma5'
# 'a1 bai4 ku4 re4 si1 ma5'

if __name__ == '__main__':
    sentences = [
        'su1 yong3 ma5'
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='pretrained/')
    parser.add_argument('--mel_save_dir', default='tacotron-output')
    parser.add_argument('--mel_load_dir', default='pretrained/ckpt-5575k.pt')
    parser.add_argument('--wav_save_dir', default='melgan-output')
    args = parser.parse_args()

    mel_save_dir = os.path.join(args.mel_save_dir)
    os.makedirs(mel_save_dir, exist_ok=True)
    wav_save_dir = os.path.join(args.wav_save_dir)
    os.makedirs(wav_save_dir, exist_ok=True)

    # modified_hp = hparams.parse(args.hparams)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # run_eval(args,sentences)
    # ckpt = torch.load(args.mel_load_dir, map_location=torch.device('cpu'))
    # vocoder = Generator(80)
    # vocoder = vocoder    #
    # vocoder.load_state_dict(ckpt['G'])
    # testset = glob.glob(os.path.join(args.mel_save_dir, '*.mel'))
    # for i, test_path in enumerate(tqdm(testset)):
    #     mel = torch.load(test_path)
    #     mel = mel.unsqueeze(0)
    #     g_audio = vocoder(mel)
    #     g_audio = g_audio.squeeze().cpu()
    #     audio = (g_audio.detach().numpy() * 32768)
    #     audio = audio[:find_endpoint(audio)]
    #     g_spec = librosa.stft(y=audio, n_fft=1024, hop_length=256, win_length=1024)
    #     write(Path(args.wav_save_dir) / ('generated-%d.wav' % i), 22050, audio.astype('int16'))



    synth,base_path=init_tacotron(args)
    vocoder = init_melgan(args)


    for i, text in enumerate(tqdm(sentences)):
        mel=synth.synthesize(args, text, base_path, i)
        mel = mel.unsqueeze(0)
        g_audio = vocoder(mel)
        g_audio = g_audio.squeeze().cpu()
        audio = (g_audio.detach().numpy() * 32768)
        audio = audio[:find_endpoint(audio)]
        g_spec = librosa.stft(y=audio, n_fft=1024, hop_length=256, win_length=1024)
        write(Path(args.wav_save_dir) / ('generated-%d.wav' % i), 22050, audio.astype('int16'))


