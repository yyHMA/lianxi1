import numpy as np
import tensorflow as tf
import os, re, io, argparse, torch
# from jamo import hangul_to_jamo
from hparams import hparams
# from librosa import effects
from models import create_model
from util.text import text_to_sequence, sequence_to_text
from util import plot
# from pathlib import Path


sentences = [
  'ni2 hao3 lian2 zhong4 you1 che1'
]


class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
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
    print('seq',seq)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    # print(feed_dict)
    input_seq, alignment, mel = self.session.run([self.inputs, self.alignments, self.mel_outputs], feed_dict=feed_dict)
    input_seq = sequence_to_text(input_seq)
    plot.plot_alignment(alignment, '%s-%d-align.png' % (base_path, idx), input_seq)
    mel_spectrogram = torch.from_numpy(mel.T)
    save_path = os.path.join(args.save_dir, 'mel-%d.mel' % (idx + 1))
    torch.save(mel_spectrogram, save_path)


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  synth = Synthesizer()
  # 改
  checkpoint_path = tf.train.get_checkpoint_state(args.checkpoint).model_checkpoint_path
  synth.load(checkpoint_path)

  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    synth.synthesize(args, text, base_path, i)



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default='pretrained/')
  parser.add_argument('--save_dir', default='tacotron-output')
  args = parser.parse_args()
  save_dir = os.path.join(args.save_dir)
  os.makedirs(save_dir, exist_ok=True)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  run_eval(args)


if __name__ == '__main__':
  main()
