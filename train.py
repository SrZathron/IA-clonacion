import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss
from text.symbols import symbols

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

global_step = 0

def main():
    assert torch.cuda.is_available(), "CUDA GPU is required for training."

    # Define absolute paths
    model_dir = "/content/drive/MyDrive/vits/logs/ljs_model"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "eval"), exist_ok=True)

    hps = utils.get_hparams()
    hps.model_dir = model_dir
    run(0, hps)

def run(rank, hps):
    global global_step

    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)

    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=2,  # Reducido para evitar problemas en Colab
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

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda(rank)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    optim_g = optim.AdamW(
        net_g.parameters(),
        lr=hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    optim_d = optim.AdamW(
        net_d.parameters(),
        lr=hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    try:
        g_checkpoint = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        d_checkpoint = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")

        if g_checkpoint and d_checkpoint:
            _, _, _, epoch_str = utils.load_checkpoint(g_checkpoint, net_g, optim_g)
            _, _, _, epoch_str = utils.load_checkpoint(d_checkpoint, net_d, optim_d)
            global_step = (epoch_str - 1) * len(train_loader)
        else:
            epoch_str = 1
            global_step = 0

    except FileNotFoundError:
        epoch_str = 1
        global_step = 0

    scheduler_g = optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scheduler_d = optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

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

    net_g.train()
    net_d.train()

    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
        x, x_lengths = x.cuda(non_blocking=True), x_lengths.cuda(non_blocking=True)
        spec, spec_lengths = spec.cuda(non_blocking=True), spec_lengths.cuda(non_blocking=True)
        y, y_lengths = y.cuda(non_blocking=True), y_lengths.cuda(non_blocking=True)

        with autocast(device_type="cuda", enabled=hps.train.fp16_run):
            y_hat, *_ = net_g(x, x_lengths, spec, spec_lengths)
            loss_g = generator_loss(y_hat, y)

        scaler.scale(loss_g).backward()
        scaler.step(optim_g)
        scaler.update()
        optim_g.zero_grad()

        logger.info(f"Train Step {global_step}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss_g.item()}")
        writer.add_scalar("Loss/train", loss_g.item(), global_step)

        global_step += 1

    net_g.eval()
    net_d.eval()

    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
            x, x_lengths = x.cuda(non_blocking=True), x_lengths.cuda(non_blocking=True)
            spec, spec_lengths = spec.cuda(non_blocking=True), spec_lengths.cuda(non_blocking=True)
            y, y_lengths = y.cuda(non_blocking=True), y_lengths.cuda(non_blocking=True)

            with autocast(device_type="cuda", enabled=hps.train.fp16_run):
                y_hat, *_ = net_g(x, x_lengths, spec, spec_lengths)

            logger.info(f"Eval Step {global_step}, Epoch {epoch}, Batch {batch_idx}")
            writer_eval.add_scalar("Loss/eval", loss_g.item(), global_step)

if __name__ == "__main__":
    main()
