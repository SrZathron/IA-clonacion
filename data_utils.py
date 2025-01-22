import os
import random
import torch
import torchaudio
from torchaudio.transforms import Resample
from utils import load_filepaths_and_text
from mel_processing import spectrogram_torch
from text import text_to_sequence, cleaned_text_to_sequence
import commons


class TextAudioLoader(torch.utils.data.Dataset):
    """
    1) Loads pairs of audio and text.
    2) Normalizes the text and converts it to sequences of integers.
    3) Computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
        self.use_cached_specs = getattr(hparams, "use_cached_specs", True)
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        """
        Filters text and stores spectrogram lengths.
        """
        audiopaths_and_text_new = []
        lengths = []

        for audiopath, text in self.audiopaths_and_text:
            # Check if the file exists
            if not os.path.exists(audiopath):
                print(f"[ERROR] File not found: {audiopath}")
                continue

            # Check text length
            if not (self.min_text_len <= len(text) <= self.max_text_len):
                print(f"[DISCARDED] Text out of range: '{text}' (length: {len(text)})")
                continue

            try:
                # Compute spectrogram length
                spec_length = os.path.getsize(audiopath) // (2 * self.hop_length)
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(spec_length)
            except Exception as e:
                print(f"[ERROR] Failed to process file {audiopath}: {e}")

        if not audiopaths_and_text_new:
            print("[CRITICAL ERROR] No valid data after filtering.")
            raise ValueError("Dataset is empty after filtering.")

        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths
        print(f"[INFO] Valid data loaded: {len(self.audiopaths_and_text)} items.")

    def get_audio_text_pair(self, audiopath_and_text):
        # Split filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            print(f"[INFO] Resampling {filename} from {sampling_rate} Hz to {self.sampling_rate} Hz.")
            resample = Resample(sampling_rate, self.sampling_rate)
            audio = resample(audio)

        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)

        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_cached_specs and os.path.exists(spec_filename):
            spec = torch.load(spec_filename, weights_only=True)
        else:
            spec = spectrogram_torch(
                audio_norm, self.filter_length, self.sampling_rate,
                self.hop_length, self.win_length, center=False
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """Zero-pads model inputs and targets to align them."""
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """
        Combines text and audio batches for training.

        batch: [(text_normalized, spec_normalized, wav_normalized), ...]
        """
        if not batch:
            raise ValueError("[ERROR] Empty batch encountered during collation.")

        # Sort by spectrogram length (descending)
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max(len(x[0]) for x in batch)
        max_spec_len = max(x[1].size(1) for x in batch)
        max_wav_len = max(x[2].size(1) for x in batch)

        text_padded = torch.LongTensor(len(batch), max_text_len).zero_()
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len).zero_()
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len).zero_()

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing

        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


def load_wav_to_torch(filename):
    """
    Load a WAV file into a PyTorch tensor.
    """
    audio, sampling_rate = torchaudio.load(filename)
    return audio.squeeze(0), sampling_rate
