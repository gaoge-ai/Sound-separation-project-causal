import os
import argparse
import torch
import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from DNN_models.Complex_MTASS_model import ComplexMTASSLightning
from DNN_models.Complex_MTASS import *
from DNN_models.Complex_MTASS_Solver import *
from online_mix_dataset import OnlineMixDataset

class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = None
        with h5py.File(h5_path, 'r') as f:
            self.length = f['X1'].shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        X1 = torch.from_numpy(self.h5_file['X1'][idx]).float()
        Y_targets = [torch.from_numpy(self.h5_file[f'Y{i}'][idx]).float() for i in range(1, 4)]
        R_targets = [torch.from_numpy(self.h5_file[f'R{i}'][idx]).float() for i in range(1, 4)]
        
        return (X1, *Y_targets, *R_targets)
    
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

def main(args):
    pl.seed_everything(42)
    l1_loss_weight = args.l1_loss_weight
    if l1_loss_weight is None:
        l1_loss_weight = 1.0 if args.use_l1_loss else 0.0
    magnitude_l1_loss_weight = args.magnitude_l1_loss_weight

    # Load dataset
    data_train, data_val = build_datasets(args)
    train_loader = DataLoader(data_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.n_workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(data_val,
                            batch_size=args.eval_batch_size,
                            shuffle=False,
                            num_workers=args.n_workers,
                            pin_memory=True,
                            drop_last=True)
    
    model = ComplexMTASSLightning(
        learning_rate=args.lr,
        model_class=Complex_MTASS,
        loss_class=Complex_MTASS_model,
        mse_loss_weight=args.mse_loss_weight,
        sisdr_loss_weight=args.sisdr_loss_weight,
        l1_loss_weight=l1_loss_weight,
        magnitude_l1_loss_weight=magnitude_l1_loss_weight,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, 'checkpoints'),
        filename="{epoch:04d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
    )

    logger = TensorBoardLogger(args.exp_dir, name="runs")

    trainer = pl.Trainer(
        default_root_dir=args.exp_dir,
        devices=args.gpus if args.use_cuda else "auto",
        accelerator="gpu" if args.use_cuda else "cpu",
        benchmark=True,
        strategy="ddp" if args.use_cuda else "auto",
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=20.0 if args.gradient_clip else 0.0,
        precision='32',
        #log_every_n_steps=10,
    )

    ckpt_path = None
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt_path = args.resume_ckpt
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)


def build_datasets(args):
    if args.data_mode == 'h5':
        return HDF5Dataset(args.train_h5), HDF5Dataset(args.val_h5)

    missing_args = [
        arg_name
        for arg_name, value in (
            ('--train_source_csv', args.train_source_csv),
            ('--val_source_csv', args.val_source_csv),
        )
        if value is None
    ]
    if missing_args:
        raise ValueError(
            "--data_mode online_csv requires " + ", ".join(missing_args)
        )

    common_kwargs = dict(
        audio_root=args.audio_root,
        num_sources_choices=args.online_num_sources,
        num_sources_probs=args.online_num_sources_probs,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        target_duration=args.target_duration,
        seed=args.online_seed,
        rir_root=args.rir_root,
        rir_prob=args.rir_prob,
        rir_room_probs=args.rir_room_probs,
    )
    data_train = OnlineMixDataset(
        source_csv=args.train_source_csv,
        samples_per_epoch=args.train_samples_per_epoch,
        deterministic=False,
        **common_kwargs,
    )
    data_val = OnlineMixDataset(
        source_csv=args.val_source_csv,
        samples_per_epoch=args.val_samples_per_epoch,
        deterministic=True,
        **common_kwargs,
    )
    return data_train, data_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, default='./model_parameters')
    parser.add_argument('--data_mode', type=str, default='h5', choices=['h5', 'online_csv'],
                        help="Dataset mode: h5 uses pre-extracted features, online_csv mixes raw audio online")
    parser.add_argument('--train_h5', type=str, default='/ssd2.m2/sound/VGGSound/imagebind/train_ready.h5')
    parser.add_argument('--val_h5', type=str, default='/ssd2.m2/sound/VGGSound/imagebind/dev_ready.h5')
    parser.add_argument('--train_source_csv', type=str, default=None,
                        help="Single-category source CSV for online training")
    parser.add_argument('--val_source_csv', type=str, default=None,
                        help="Single-category source CSV for deterministic online validation")
    parser.add_argument('--audio_root', type=str, default=None,
                        help="Root directory for relative audio paths in online source CSVs")
    parser.add_argument('--train_samples_per_epoch', type=int, default=20000,
                        help="Number of online mixtures sampled for each training epoch")
    parser.add_argument('--val_samples_per_epoch', type=int, default=5000,
                        help="Number of deterministic online mixtures sampled for each validation epoch")
    parser.add_argument('--online_num_sources', nargs='+', type=int, default=[2, 3],
                        help="Possible number of unique-category sources per online mixture")
    parser.add_argument('--online_num_sources_probs', nargs='+', type=float, default=None,
                        help="Sampling weights for --online_num_sources; defaults to uniform")
    parser.add_argument('--snr_min', type=float, default=-3.0,
                        help="Minimum SNR in dB for non-reference online sources")
    parser.add_argument('--snr_max', type=float, default=3.0,
                        help="Maximum SNR in dB for non-reference online sources")
    parser.add_argument('--target_duration', type=float, default=10.0,
                        help="Online audio duration in seconds")
    parser.add_argument('--online_seed', type=int, default=42,
                        help="Seed used for online validation sampling")
    parser.add_argument('--rir_root', type=str, default=None,
                        help="Root directory of RIR files; disabled by default")
    parser.add_argument('--rir_prob', type=float, default=0.0,
                        help="Probability of applying RIR to eligible online sources")
    parser.add_argument('--rir_room_probs', nargs='+', type=float, default=None,
                        help="Sampling weights for small/medium/large RIR rooms; defaults to uniform")
    parser.add_argument('--resume_ckpt', type=str, default=None, help="Path to .ckpt to continue training")
    parser.add_argument("--gpus", nargs="+", type=int, help="e.g. --gpus 0 1 2")

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--gradient_clip', action='store_true', help="Enable gradient clipping")
    parser.add_argument('--mse_loss_weight', type=float, default=1.0, help="Weight for MSE loss; 0 disables it")
    parser.add_argument('--sisdr_loss_weight', type=float, default=1.0, help="Weight for SI-SDR loss; 0 disables it")
    parser.add_argument('--l1_loss_weight', type=float, default=None, help="Weight for L1 loss; 0 disables it")
    parser.add_argument('--magnitude_l1_loss_weight', type=float, default=0.0, help="Weight for magnitude L1 loss; 0 disables it")
    parser.add_argument('--use_l1_loss', action='store_true', help="Deprecated compatibility flag; enables L1 with weight 1.0 if --l1_loss_weight is not set")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")

    args = parser.parse_args()

    if args.gpus is not None:
        args.use_cuda = True

    os.makedirs(args.exp_dir, exist_ok=True)
    
    main(args)
