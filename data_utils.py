import os
import torch
import librosa
import numpy as np
from utils import logger
from torch.utils.data import Dataset
from unidecode import unidecode

class TextAudioLoader(Dataset):  # <<< ¡Clase CORRECTA!
    def __init__(self, filelist_path, hparams):
        self.hparams = hparams
        self.audiopaths = []
        self.texts = []
        
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) != 2:
                    continue
                path, text = parts
                full_path = os.path.join("/content", path.strip())
                if os.path.exists(full_path):
                    self.audiopaths.append(full_path)
                    self.texts.append(self.normalize_text(text))
                else:
                    logger.warning(f"⚠️ Archivo no encontrado: {full_path}")
        logger.info(f"✅ Dataset cargado: {len(self.audiopaths)} muestras válidas")

    def normalize_text(self, text):
        """Normalización avanzada para español"""
        text = unidecode(text).lower()
        replacements = {
            '¡': '', '!': '.', '¿': '', '?': '.',
            '\n': ' ', '\t': ' ', '  ': ' ', '--': '-', '“': '"', '”': '"'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text[:self.hparams['data'].get('max_text_len', 300)]

    def __getitem__(self, idx):
        audio, sr = librosa.load(
            self.audiopaths[idx], 
            sr=self.hparams['data']['sampling_rate'],
            mono=True
        )
        return torch.FloatTensor(audio), self.texts[idx]

    def __len__(self):
        return len(self.audiopaths)

class TextAudioCollate:  # <<< Clase única y corregida
    def __init__(self, hparams):
        self.hparams = hparams
        self.symbols = hparams['data']['symbols']  # ¡Acceso correcto!

    def __call__(self, batch):
        target_sr = self.hparams['data']['sampling_rate']
        max_audio_len = self.hparams['data'].get('max_wav_length', target_sr * 45)
        
        audios = []
        audio_lengths = []
        texts = []
        
        # Procesamiento de audio
        for audio, text in batch:
            if audio.size(0) > max_audio_len:
                start = torch.randint(0, audio.size(0) - max_audio_len, (1,)).item()
                audio = audio[start:start + max_audio_len]
            else:
                pad = (0, max_audio_len - audio.size(0))
                audio = torch.nn.functional.pad(audio, pad, mode='constant', value=0.0)
            
            audios.append(audio)
            audio_lengths.append(audio.size(0))
            texts.append(text)

        # Procesamiento de texto
        text_ids = []
        text_lengths = []
        for text in texts:
            text_id = [self.symbols.index(c) for c in text if c in self.symbols]
            text_ids.append(torch.tensor(text_id, dtype=torch.long))
            text_lengths.append(len(text_id))
        
        # Padding de textos
        text_ids = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True)
        text_lengths = torch.tensor(text_lengths, dtype=torch.long)
        
        return (
            torch.stack(audios),
            torch.tensor(audio_lengths, dtype=torch.long),
            text_ids,
            text_lengths
        )