import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss
from text.symbols import symbols

torch.backends.cudnn.benchmark = True
global_step = 0

def main():
    assert torch.cuda.is_available(), "CUDA GPU is required for training."

    # Configurar rutas absolutas
    model_dir = "/content/drive/MyDrive/vits/logs/ljs_model"
    os.makedirs(model_dir, exist_ok=True)  # Crear directorio si no existe
    os.makedirs(os.path.join(model_dir, "eval"), exist_ok=True)  # Crear subdirectorio eval si no existe

    hps = utils.get_hparams()
    hps.model_dir = model_dir
    run(0, hps)

def run(rank, hps):
    global global_step

    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)

    # Configuración de SummaryWriter con rutas absolutas
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # Prueba de escritura de logs
    writer.add_scalar("test_scalar", 1, 0)
    writer.close()
    print("Prueba de logs completada.")

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    # Preparar datasets
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=TextAudioCollate(),
    )
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=hps.train.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=TextAudioCollate(),
    )

    # Verificar la cantidad de datos cargados
    print(f"Cantidad de datos de entrenamiento: {len(train_loader)}")
    print(f"Cantidad de datos de evaluación: {len(eval_loader)}")

    # Inicializar modelos
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # Manejo de checkpoints
    try:
        latest_g_path = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        latest_d_path = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")

        if latest_g_path and latest_d_path:
            _, _, _, epoch_str = utils.load_checkpoint(latest_g_path, net_g, optim_g)
            _, _, _, epoch_str = utils.load_checkpoint(latest_d_path, net_d, optim_d)
            global_step = (epoch_str - 1) * len(train_loader)
        else:
            print("[INFO] No se encontraron checkpoints previos. Iniciando desde cero.")
            epoch_str = 1
            global_step = 0

    except FileNotFoundError:
        print("[ERROR] Archivo de checkpoint no encontrado. Iniciando desde cero.")
        epoch_str = 1
        global_step = 0

    # Configuración de schedulers
    scheduler_g = optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    # Entrenamiento y evaluación
    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(
            epoch,
            hps,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            scaler,
            [train_loader, eval_loader],
            logger,
            [writer, writer_eval],
        )
        scheduler_g.step()
        scheduler_d.step()

def train_and_evaluate(epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    global global_step
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    writer, writer_eval = writers

    # Entrenamiento
    net_g.train()
    net_d.train()
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
        print(f"Inicio del Epoch {epoch}, Batch {batch_idx}")
        x, x_lengths = x.cuda(0, non_blocking=True), x_lengths.cuda(0, non_blocking=True)
        spec, spec_lengths = spec.cuda(0, non_blocking=True), spec_lengths.cuda(0, non_blocking=True)
        y, y_lengths = y.cuda(0, non_blocking=True), y_lengths.cuda(0, non_blocking=True)

        try:
            with autocast(device_type="cuda", enabled=hps.train.fp16_run):
                y_hat, *_ = net_g(x, x_lengths, spec, spec_lengths)
                loss_g = generator_loss(y_hat, y)
                loss_value = loss_g.item()

            print(f"Training - Epoch: {epoch}, Batch: {batch_idx}, Global Step: {global_step}")
            print(f"Loss/train: {loss_value}")

            writer.add_scalar("Loss/train", loss_value, global_step)
            writer.flush()  # Forzar escritura inmediata

            global_step += 1

        except Exception as e:
            print(f"Error durante el entrenamiento en Batch {batch_idx}: {e}")

    # Evaluación
    net_g.eval()
    net_d.eval()
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
            print(f"Inicio del bucle de evaluación - Epoch: {epoch}, Batch: {batch_idx}")
            x, x_lengths = x.cuda(0, non_blocking=True), x_lengths.cuda(0, non_blocking=True)
            spec, spec_lengths = spec.cuda(0, non_blocking=True), spec_lengths.cuda(0, non_blocking=True)
            y, y_lengths = y.cuda(0, non_blocking=True), y_lengths.cuda(0, non_blocking=True)

            try:
                with autocast(device_type="cuda", enabled=hps.train.fp16_run):
                    y_hat, *_ = net_g(x, x_lengths, spec, spec_lengths)
                    eval_loss = generator_loss(y_hat, y)
                    eval_loss_value = eval_loss.item()

                print(f"Evaluation - Epoch: {epoch}, Batch: {batch_idx}, Global Step: {global_step}")
                print(f"Loss/eval: {eval_loss_value}")

                writer_eval.add_scalar("Loss/eval", eval_loss_value, global_step)
                writer_eval.flush()  # Forzar escritura inmediata

            except Exception as e:
                print(f"Error durante la evaluación en Batch {batch_idx}: {e}")

if __name__ == "__main__":
    main()
