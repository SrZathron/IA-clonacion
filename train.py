import os
import torch
import time
from torch import optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Configuración inicial
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Módulos personalizados
from utils import setup_logger, get_hparams, load_checkpoint, save_checkpoint
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss
from text.symbols import symbols

# Rutas absolutas
CONFIG_PATH = "/content/drive/MyDrive/vits/configs/ljs_base.json"
MODEL_DIR = "/content/drive/MyDrive/vits/logs/ljs_model"
FILELIST_TRAIN = "/content/drive/MyDrive/vits/filelists/ljs_audio_text_train_filelist.txt.cleaned"
FILELIST_VAL = "/content/drive/MyDrive/vits/filelists/ljs_audio_text_val_filelist.txt.cleaned"

logger = setup_logger()

def main():
    try:
        # Configuración
        hparams = get_hparams()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Modelos
        net_g = SynthesizerTrn(
            len(symbols),
            hparams['data']['filter_length'] // 2 + 1,
            hparams['train']['segment_size'] // hparams['data']['hop_length'],
            **hparams['model']
        ).to(device)
        
        net_d = MultiPeriodDiscriminator(hparams['model']['use_spectral_norm']).to(device)
        
        # Optimizadores
        optim_g = optim.AdamW(net_g.parameters(), 
                            lr=hparams['train']['learning_rate'],
                            betas=hparams['train']['betas'],
                            eps=hparams['train']['eps'])
        
        optim_d = optim.AdamW(net_d.parameters(), 
                             lr=hparams['train']['learning_rate'],
                             betas=hparams['train']['betas'],
                             eps=hparams['train']['eps'])
        
        # Cargar checkpoints
        start_epoch = 0
        if os.path.exists(os.path.join(MODEL_DIR, "G_latest.pth")):
            net_g, optim_g, start_epoch = load_checkpoint(os.path.join(MODEL_DIR, "G_latest.pth"), net_g, optim_g)
            net_d, optim_d, _ = load_checkpoint(os.path.join(MODEL_DIR, "D_latest.pth"), net_d, optim_d)
            logger.info("✅ Checkpoints cargados")
        else:
            logger.info("🚀 Iniciando entrenamiento desde cero")

        # Datasets y collate
        collate_fn = TextAudioCollate(hparams)
        train_dataset = TextAudioLoader(FILELIST_TRAIN, hparams)
        train_loader = DataLoader(
            train_dataset,
            batch_size=hparams['train']['batch_size'],
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )

        # Componentes de entrenamiento
        scaler = GradScaler(enabled=hparams['train']['fp16_run'])
        writer = SummaryWriter(log_dir=MODEL_DIR) if os.path.exists(MODEL_DIR) else None
        
        # Ciclo de entrenamiento
        for epoch in range(start_epoch, hparams['train']['epochs']):
            net_g.train()
            net_d.train()
            epoch_start = time.time()
            
            for batch_idx, (audios, audio_lengths, text_ids, text_lengths) in enumerate(train_loader):
                audios = audios.to(device)
                text_ids = text_ids.to(device)
                text_lengths = text_lengths.to(device)
                audio_lengths = audio_lengths.to(device)
                
                # Forward CORRECTO
                y_hat, l_length, attn, ids_slice = net_g(
                    x=text_ids, 
                    x_lengths=text_lengths,
                    y=audios,
                    y_lengths=audio_lengths
                )                    
                loss_g = generator_loss(y_hat, audios) + l_length
                
                scaler.scale(loss_g).backward()
                scaler.step(optim_g)
                scaler.update()
                optim_g.zero_grad(set_to_none=True)

                # Entrenamiento del Discriminador
                with autocast(device_type="cuda", enabled=hparams['train']['fp16_run']):
                    y_d_hat_r, y_d_hat_g, _ = net_d(audios, y_hat.detach())
                    loss_d = discriminator_loss(y_d_hat_r, y_d_hat_g)
                
                scaler.scale(loss_d).backward()
                scaler.step(optim_d)
                scaler.update()
                optim_d.zero_grad(set_to_none=True)

                # Logging
                if batch_idx % 50 == 0:
                    elapsed = time.time() - epoch_start
                    samples_sec = (batch_idx + 1) * hparams['train']['batch_size'] / elapsed
                    logger.info(
                        f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss G: {loss_g.item():.4f} | Loss D: {loss_d.item():.4f} | "
                        f"Speed: {samples_sec:.1f} samples/s"
                    )
                    
                    if writer:
                        global_step = epoch * len(train_loader) + batch_idx
                        writer.add_scalar("Loss/Generator", loss_g.item(), global_step)
                        writer.add_scalar("Loss/Discriminator", loss_d.item(), global_step)

            # Guardado de checkpoint
            save_checkpoint(net_g, optim_g, epoch+1, os.path.join(MODEL_DIR, f"G_{epoch+1}.pth"))
            save_checkpoint(net_d, optim_d, epoch+1, os.path.join(MODEL_DIR, f"D_{epoch+1}.pth"))
            logger.info(f"💾 Checkpoint época {epoch+1} guardado (Tiempo: {time.time()-epoch_start:.1f}s)")

    except Exception as e:
        logger.error(f"🚨 Error crítico: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()