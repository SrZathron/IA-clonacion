class HParams:
    def __init__(self):
        # Configuración de entrenamiento
        self.train = {
            "log_interval": 200,
            "eval_interval": 1000,
            "seed": 1234,
            "epochs": 20000,
            "learning_rate": 0.0002,
            "betas": [0.8, 0.99],
            "eps": 1e-09,
            "batch_size": 64,
            "fp16_run": True,
            "lr_decay": 0.999875,
            "segment_size": 8192,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
        }

        # Configuración de datos
        self.data = {
            "training_files": "/content/drive/MyDrive/vits/filelists/ljs_audio_text_train_filelist.txt.cleaned",
            "validation_files": "/content/drive/MyDrive/vits/filelists/ljs_audio_text_val_filelist.txt.cleaned",
            "text_cleaners": ["spanish_cleaners"],
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "add_blank": True,
            "n_speakers": 0,
            "cleaned_text": True,
            "min_text_len": 1,
            "max_text_len": 700,
        }

        # Configuración del modelo
        self.model = {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5],
            ],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "n_layers_q": 3,
            "use_spectral_norm": False,
        }

        # Directorio del modelo
        self.model_dir = "./logs/ljs_model"


# Instancia de configuración
hps = HParams()
