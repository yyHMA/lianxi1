import numpy as np
import os, argparse, glob, librosa, librosa.display, torch, scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
from gan import Generator
from pathlib import Path
from scipy.io.wavfile import write


def main(args):
    vocoder = Generator(80)
    vocoder = vocoder.cuda()
    ckpt = torch.load(args.load_dir)
    vocoder.load_state_dict(ckpt['G'])
    testset = glob.glob(os.path.join(args.test_dir, '*.mel'))
    for i, test_path in enumerate(tqdm(testset)):
        mel = torch.load(test_path)
        mel = mel.unsqueeze(0)
        g_audio = vocoder(mel.cuda())
        g_audio = g_audio.squeeze().cpu()
        audio = (g_audio.detach().numpy() * 32768)
        audio = audio[:find_endpoint(audio)]
        g_spec = librosa.stft(y=audio, n_fft=1024, hop_length=256, win_length=1024)
        write(Path(args.save_dir) / ('generated-%d.wav' % i), 22050, audio.astype('int16'))
        plot_stft(g_spec, i)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(22050 * min_silence_sec)
  hop_length = window_length // 4
  threshold = np.power(10.0, threshold_db * 0.05) * 32768
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x : x + window_length]) < threshold:
      return x + hop_length
  return len(wav)


def plot_stft(g_spec, idx):
    plt.figure(figsize=(12, 4))
    spectrogram = librosa.amplitude_to_db(np.abs(g_spec), ref=np.max)
    librosa.display.specshow(spectrogram, x_axis='time', y_axis='log', hop_length=256)
    plt.title('generated audio spectrogram')
    plt.tight_layout()
    fn = 'spectrogram-%d.png' % idx
    plt.savefig(args.save_dir + '/' + fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default='tacotron-output')
    parser.add_argument('--load_dir', required=True)
    parser.add_argument('--save_dir', default='melgan-output')
    args = parser.parse_args()
    save_dir = os.path.join(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    main(args)
