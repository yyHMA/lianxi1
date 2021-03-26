import numpy as np
import argparse, os, re, librosa
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm
from hparams import hparams1 as hp
from util import audio


# text_name = 'transcript.v.1.2.txt'
text_name = 'metadata.csv'
filters = "([.,!?])"


def preprocess_kss(args):
  in_dir = os.path.join(args.base_dir, 'BZNSYP')
  out_dir = os.path.join(args.base_dir, args.output)
  #out_dir=args.output
  os.makedirs(out_dir, exist_ok=True)
  metadata = build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1
  with open(os.path.join(in_dir, text_name), encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('|')
      wav_path = os.path.join(in_dir, 'wavs',parts[0]+'.wav')
      text = parts[1]
      text = re.sub(re.compile(filters), '', text)
      futures.append(executor.submit(_process_utterance, out_dir, index, wav_path, text))
      index += 1
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
  wav, sr = librosa.core.load(wav_path, sr=hp.sample_rate)
  mel_basis = librosa.filters.mel(sr, hp.n_fft, hp.num_mels)
  spectrogram = librosa.stft(y=wav, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
  n_frames = spectrogram.shape[1]
  mel_spectrogram = np.dot(mel_basis, np.abs(spectrogram)).astype(np.float32)
  mel_filename = 'kss-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
  return (mel_filename, n_frames, text)


def write_metadata(metadata, out_dir):
  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')
  frames = sum([m[1] for m in metadata])
  print('Wrote %d utterances, %d frames' % (len(metadata), frames))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('/ssd-data/data'))
  parser.add_argument('--output', default='tacotron-train')
  parser.add_argument('--num_workers', type=int, default=cpu_count())
  args = parser.parse_args()
  preprocess_kss(args)


if __name__ == "__main__":
  main()
