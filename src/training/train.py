import datetime
import os
import time

import numpy as np
import torch
from tqdm import tqdm

import wandb
from config.dataset import MTATConfig
from config.full import GlobalConfig
from config.train import TrainConfig
from src.model.model import Siamese


class Trainer:
    def __init__(
        self,
        global_config=GlobalConfig(),
        override_wandb=True,
        debug=False,
        resume_id=None,
    ) -> None:
        self.global_config = global_config
        self.config = TrainConfig(dict=global_config.train_config)
        self.dataset_config = MTATConfig(dict=global_config.MTAT_config)
        self.debug = debug
        self.first_run = True
        if (self.config.log_wandb or override_wandb) and override_wandb:
            if resume_id is None:
                self.wandb_run = wandb.init(
                    project="EquiTempo", config=global_config.to_dict()
                )
            else:
                print(f"resuming run {resume_id}")
                self.wandb_run = wandb.init(
                    project="EquiTempo",
                    config=global_config.to_dict(),
                    resume="must",
                    id=resume_id,
                )
            self.wandb_run_name = self.wandb_run.name

        else:
            self.wandb_run = None
            current_time = datetime.datetime.now()
            self.wandb_run_name = current_time.strftime("%d-%m-%H%M%S")
        self.it = 0
        self.epoch = 0

    def init_model(self, path=None, test=False, override_device=None):
        device = self.config.device
        if override_device is not None:
            device = override_device
        model = Siamese(
            filters=self.config.filters,
            dilations=self.config.dilations,
            dropout_rate=self.config.dropout_rate,
            output_dim=self.config.output_dim,
        ).to(device)

        original_state_dict = model.state_dict()

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr, betas=(0.9, 0.999)
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
        it = self.it
        if path is not None:
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint["gen_state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.debug:
                for name, param in model.named_parameters():
                    if name in original_state_dict:
                        # Check if the weights match
                        if torch.equal(param.data, original_state_dict[name].data):
                            print(f"Weights for layer '{name}' are the same.")
                        else:
                            print(f"Weights for layer '{name}' are different.")
                    else:
                        print(
                            f"Layer '{name}' not found in the loaded state dictionary."
                        )

            it = checkpoint["it"]
            epoch = 0
            if "epoch" in checkpoint:
                epoch = checkpoint["epoch"]
                self.epoch = epoch
            self.it = it
        if test:
            model.eval()
        return model, optimizer, scaler, self.it, self.epoch

    def save_model(self, loss, it, model, optimizer, scaler):
        os.makedirs(self.config.save_path + f"/{self.wandb_run_name}", exist_ok=True)

        torch.save(
            {
                "gen_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": loss,
                "epoch": self.epoch,
                "it": it,
            },
            "random.pt",
        )

    def save_config(self):
        os.makedirs(self.config.save_path + f"/{self.wandb_run_name}", exist_ok=True)
        self.global_config.save(
            self.config.save_path + f"/{self.wandb_run_name}/config.yml"
        )

    def loss_function(self, c1, c2, alpha1, alpha2, eps=1e-7):
        c_ratio = c1 / (c2 + eps)
        alpha_ratio = alpha1 / (alpha2 + eps)
        if self.first_run:
            print("c_ratio:", c_ratio.shape)
            print("alpha_ratio:", alpha_ratio.shape)
        return torch.abs(c_ratio.squeeze() - alpha_ratio.squeeze()).mean()

    def train_iteration(self, x1, x2, alpha1, alpha2, model, optimizer, scaler):
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.config.mixed_precision
        ):
            if self.first_run:
                print("x1:", x1.shape)
                print("x2:", x2.shape)
            _, c1 = model(x1)
            _, c2 = model(x2)
            if self.first_run:
                print("c1:", c1.shape)
                print("c2:", c2.shape)
                print("alpha1:", alpha1.shape)
                print("alpha2:", alpha2.shape)
            loss1 = self.loss_function(c1, c2, alpha1, alpha2)
            loss2 = self.loss_function(c2, c1, alpha2, alpha1)
        loss = 0.5 * (loss1 + loss2)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        self.first_run = False
        return loss.item()

    def update_lr(self, new_lr, optimizer):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def train_loop(self, dataloader, model, optimizer, scaler, it=0, writer=None):
        self.save_config()

        dataloader_length = len(dataloader)
        if self.config.warmup:
            self.update_lr(1e-7, optimizer)
            target_lr = self.config.lr
            lr = 1e-7
        else:
            self.update_lr(self.config.lr, optimizer)
        model = model.to(self.config.device)
        model.train()

        try:
            counter = 0
            loss = 0.0
            for epoch in range(self.epoch, self.config.epochs):
                bef = time.time()
                bef_loop = time.time()
                loss_list = []
                pbar = tqdm(
                    dataloader,
                    desc=f"Epoch {epoch}/{self.config.epochs}",
                    leave=True,
                    total=dataloader_length,
                )
                for batch_i, data in enumerate(pbar):
                    loss = self.train_iteration(
                        data["audio_1"].to(self.config.device),
                        data["audio_2"].to(self.config.device),
                        data["rp_1"].to(self.config.device),
                        data["rp_2"].to(self.config.device),
                        model,
                        optimizer,
                        scaler,
                    )
                    if writer is not None:
                        writer.add_scalar("loss", loss, it)
                        writer.add_scalar(
                            "learning_rate", optimizer.param_groups[0]["lr"], it
                        )
                    if self.wandb_run is not None and it % 10 == 0:
                        self.wandb_run.log(
                            {
                                "loss": loss,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                            },
                            step=it,
                        )
                    loss_list.append(loss)
                    counter += 1
                    it += 1
                    self.it = it
                    if self.config.warmup:
                        if epoch == 0:
                            lr = lr + (1 / dataloader_length) * target_lr
                            self.update_lr(lr, optimizer)

                    if batch_i % self.config.display_progress_every == 0:
                        pbar.set_postfix(
                            {
                                "Loss_sc": np.mean(loss_list[-counter:], axis=0),
                                "Iter": it,
                                "LR": optimizer.param_groups[0]["lr"],
                                "Time/Iter": (time.time() - bef_loop)
                                / self.config.display_progress_every,
                            }
                        )
                        bef_loop = time.time()
                self.save_model(
                    np.mean(loss_list[-counter:], axis=0), it, model, optimizer, scaler
                )
                counter = 0
                self.epoch += 1

        except Exception as e:
            print(e)
        finally:
            self.save_model(
                np.mean(loss_list[-counter:], axis=0), it, model, optimizer, scaler
            )
            return it
