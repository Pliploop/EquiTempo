import numpy as np
import torch
import wandb
from learn2learn.algorithms import MAML

from config.finetune import FinetuningConfig
from config.full import GlobalConfig
from src.data_loading.datasets import MetaDataset
from src.evaluation.metrics import compute_accuracy_1, compute_accuracy_2
from src.finetuning.finetune_loss import XentBoeck
from src.training.train import Trainer


def meta_train(global_config=None):
    """Meta-train a tempo estimator on GTZAN. Meta tasks are
    created based on genre, and contain one example per class,
    which changes per iteration."""

    if global_config:
        config = FinetuningConfig(dict=global_config.finetuning_config)
    else:
        config = FinetuningConfig(dict=GlobalConfig().finetuning_config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )

    if config.wandb_log:
        wandb_run = wandb.init(project="EquiTempo", config=global_config.to_dict())
        wandb_run.name = wandb_run.name + "_meta_train"
    else:
        wandb_run = None
        wandb_run_name = ""

    model, optimizer, scaler, it, epoch = Trainer(
        override_wandb=config.wandb_log, global_config=global_config
    ).init_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=config.config_lr)

    if config.wandb_log:
        wandb.watch(models=model, log="all", log_freq=5)

    model.to(device)

    loss_function = XentBoeck(device=device)

    gtzan_genres = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]

    meta_datasets = [MetaDataset(genre=genre) for genre in gtzan_genres]
    meta_train_loaders = [dataset.create_dataloader() for dataset in meta_datasets]
    # since there's a lot of randomness in which chunk you sample for a given tempo
    # class, we'll use the same dataset for validation. Overlap should be small
    # enough to make this solution good enough.
    meta_val_loaders = [dataset.create_dataloader() for dataset in meta_datasets]

    maml = MAML(model, lr=config.lr, first_order=False)

    # generate a list of ints from 0 to len(gtzan_genres) - 1
    idx_list = list(range(len(gtzan_genres)))

    for epoch in range(config.epochs):
        # shuffle the list of ints
        new_idx_list = np.random.shuffle(idx_list)

        for task_id in new_idx_list:
            train_loader = meta_train_loaders[task_id]
            val_loader = meta_val_loaders[task_id]

            for batch in train_loader:
                learner = maml.clone()

                audio = batch["audio"].to(device)
                tempo = batch["tempo"].to(device)
                final_tempos = torch.round(tempo).long()

                classification_out, _ = learner(audio)
                preds = np.argmax(classification_out.detach().cpu().numpy(), axis=1)
                truth = (final_tempos).detach().cpu().numpy()

                acc_1 = compute_accuracy_1(truth.tolist(), preds.tolist())
                acc_2 = compute_accuracy_2(truth.tolist(), preds.tolist())
                loss = loss_function(classification_out, final_tempos)

                # Meta-update the model parameters
                learner.adapt(loss)

                # Log the validation loss to wandb if logging is enabled
                if config.wandb_log:
                    wandb.log({f"{gtzan_genres[task_id]} train loss": loss.item()})

            # outer loop
            for val_batch in val_loader:
                audio_val = val_batch["audio"].to(device)
                tempo_val = val_batch["tempo"].to(device)
                final_tempos_val = torch.round(tempo_val).long()

                classification_out_val, _ = learner(audio_val)
                preds_val = np.argmax(
                    classification_out_val.detach().cpu().numpy(), axis=1
                )
                truth_val = (final_tempos_val).detach().cpu().numpy()

                acc_1_val = compute_accuracy_1(truth_val.tolist(), preds_val.tolist())
                acc_2_val = compute_accuracy_2(truth_val.tolist(), preds_val.tolist())
                meta_loss = loss_function(classification_out_val, final_tempos_val)

                meta_loss.backward()
                optimizer.step()
                meta_optimizer.zero_grad()

                if config.wandb_log:
                    wandb.log(
                        {f"{gtzan_genres[task_id]} val loss": meta_loss.item()},
                        step=it,
                    )
