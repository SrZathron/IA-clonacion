import os
import torch
import librosa
import numpy as np
from utils import logger
from torch.utils.data import Dataset
from unidecode import unidecode

class TextAudioLoader(Dataset):
    def __init__(self, filelist_path, hparams):        self.hparams = hparams
        self.audiopaths = []
        self.texts = []
        
        # Cargar rutas verificando existencia
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) != 2:
                    continue
                path, text = parts
                # Corrección clave: Usar path directamente sin concatenar
                full_path = os.path.join("/content", path.lstrip("/"))  # ¡Cambiado!
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
        # Cargar audio sin truncar
        audio, sr = librosa.load(
            self.audiopaths[idx], 
            sr=self.hparams['data']['sampling_rate'],
            mono=True
        )
        return torch.FloatTensor(audio), self.texts[idx]

    def __len__(self):
        return len(self.audiopaths)

class TextAudioCollate:
    def __init__(self, hparams):
        self.hparams = hparams
    
    def __call__(self, batch):
        # Calcular límites desde la configuración
        target_sr = self.hparams['data']['sampling_rate']
        max_allowed = self.hparams['data'].get('max_wav_length', target_sr * 45)
        
        # Determinar máxima longitud real
        actual_max = max(a[0].size(0) for a in batch)
        max_len = min(actual_max, max_allowed)
        
        audios = []
        texts = []
        
        for audio, text in batch:
            # Recorte aleatorio si supera el máximo
            if audio.size(0) > max_len:
                start = torch.randint(0, audio.size(0) - max_len, (1,)).item()
                audio = audio[start:start + max_len]
            else:
                # Padding simétrico para mejor rendimiento
                pad_left = (max_len - audio.size(0)) // 2
                pad_right = max_len - audio.size(0) - pad_left
                audio = torch.nn.functional.pad(audio, (pad_left, pad_right), mode='reflect')
            
            audios.append(audio)
            texts.append(text)
            
        return torch.stack(audios), texts