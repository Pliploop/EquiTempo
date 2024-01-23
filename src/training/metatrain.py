import os

import numpy as np
import torch
from learn2learn.algorithms import MAML
from torch import optim
from tqdm import tqdm

import wandb
from config.finetune import FinetuningConfig
from config.full import GlobalConfig
from src.data_loading.datasets import MetaDataset
from src.evaluation.metrics import compute_accuracy_1
from src.training.train import Trainer


def fast_adaptation(
    model, device, train_batch, val_batch, loss_function, adaptation_steps
):
    train_audio = train_batch["audio"].to(device)
    train_tempo = train_batch["tempo"].to(device)
    train_final_tempos = torch.round(train_tempo).long()

    val_audio = val_batch["audio"].to(device)
    val_tempo = val_batch["tempo"].to(device)
    val_final_tempos = torch.round(val_tempo).long()

    # Adaptation
    for step in range(adaptation_steps):
        classification_out, _ = model(train_audio)
        train_error = loss_function(classification_out, train_final_tempos)
        model.adapt(train_error, allow_unused=True)

    # Evaluate adapted model
    predictions, _ = model(val_audio)
    val_error = loss_function(predictions, val_final_tempos)
    val_accuracy = compute_accuracy_1(
        val_tempo.detach().cpu().numpy().tolist(), predictions.argmax(dim=1).tolist()
    )[0]  # take the results, not accuracy

    return val_error, val_accuracy


def meta_train(global_config=None, adaptation_steps=1, batch_size=25):
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

    # Create meta tasks
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

    meta_datasets = [MetaDataset(genre=genre, n_tracks=50) for genre in gtzan_genres]
    meta_train_loaders = [dataset.create_dataloader() for dataset in meta_datasets]
    # since there's a lot of randomness in which chunk you sample for a given tempo
    # class, we'll use the same dataset for validation. Overlap should be small
    # enough to make this solution good enough.
    meta_val_loaders = [dataset.create_dataloader() for dataset in meta_datasets]

    # Create model
    model, optimizer, scaler, it, epoch = Trainer(
        override_wandb=config.wandb_log, global_config=global_config
    ).init_model()
    model.to(device)
    maml = MAML(model, lr=0.001, first_order=False)
    optimizer = optim.Adam(maml.parameters(), 0.003)
    loss_function = torch.nn.CrossEntropyLoss()

    if config.wandb_log:
        wandb.watch(models=model, log="all", log_freq=5)

    # generate a list of ints from 0 to len(gtzan_genres) - 1
    idx_list = list(range(len(gtzan_genres)))

    for epoch in range(500):
        optimizer.zero_grad()
        print(f"Epoch: {epoch}")
        # shuffle the list in-place
        np.random.shuffle(idx_list)

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0

        for task_id in idx_list:
            print(f">> Task: {gtzan_genres[task_id]}")
            train_loader = meta_train_loaders[task_id]
            val_loader = meta_val_loaders[task_id]

            # for train_batch, val_batch in tqdm(zip(train_loader, val_loader)):
            #     # compute meta-training loss
            #     learner = maml.clone()
            #     val_error, val_accuracy = fast_adaptation(
            #         model=learner,
            #         device=device,
            #         train_batch=train_batch,
            #         val_batch=val_batch,
            #         loss_function=loss_function,
            #         adaptation_steps=adaptation_steps,
            #     )

            #     task_train_error += val_error.item()
            #     task_train_accuracy += val_accuracy

            # # Backpropagate errors after accumulating them across all batches of the task
            # task_train_error /= len(train_loader) / batch_size
            # task_train_accuracy /= len(train_loader) / batch_size
            # meta_train_error += task_train_error
            # meta_train_accuracy += task_train_accuracy

            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            learner = maml.clone()
            val_error, val_accuracy = fast_adaptation(
                model=learner,
                device=device,
                train_batch=train_batch,
                val_batch=val_batch,
                loss_function=loss_function,
                adaptation_steps=adaptation_steps,
            )

            val_error.backward()
            meta_train_error += val_error.item()
            meta_train_accuracy += val_accuracy

            # Compute meta-validation loss
            try:
                meta_train_loader = meta_train_loaders[task_id + 1]
                meta_val_loader = meta_val_loaders[task_id + 1]
            except IndexError:
                meta_train_loader = meta_train_loaders[task_id - 1]
                meta_val_loader = meta_val_loaders[task_id - 1]

        #     learner = maml.clone()
        #     for train_batch, val_batch in tqdm(zip(meta_train_loader, meta_val_loader)):
        #         val_error, val_accuracy = fast_adaptation(
        #             model=learner,
        #             device=device,
        #             train_batch=train_batch,
        #             val_batch=val_batch,
        #             loss_function=loss_function,
        #             adaptation_steps=adaptation_steps,
        #         )
        #         meta_val_error += val_error.item()
        #         meta_val_accuracy += val_accuracy

        # meta_train_error /= len(gtzan_genres)
        # meta_train_accuracy /= len(gtzan_genres)
        # meta_val_error /= len(gtzan_genres)
        # meta_val_accuracy /= len(gtzan_genres)
        train_batch = next(iter(meta_train_loader))
        val_batch = next(iter(meta_val_loader))
        learner = maml.clone()
        val_error, val_accuracy = fast_adaptation(
            model=learner,
            device=device,
            train_batch=train_batch,
            val_batch=val_batch,
            loss_function=loss_function,
            adaptation_steps=adaptation_steps,
        )

        meta_valid_error += val_error.item()
        meta_valid_accuracy += val_accuracy

        # Print some metrics
        print("\n")
        print("Meta Train Error", meta_train_error)
        print("Meta Train Accuracy", meta_train_accuracy)
        print("Meta Valid Error", meta_valid_error)
        print("Meta Valid Accuracy", meta_valid_accuracy)

        # averate the accumulated gradients and optimize
        for p in maml.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / len(gtzan_genres))
        optimizer.step()

        if epoch % 10 == 0:
            save_model(
                config,
                loss_function,
                it,
                model,
                optimizer,
                epoch,
                wandb_run,
            )


def save_model(
    config,
    loss,
    it,
    model,
    optimizer,
    epoch,
    wandb_run,
):
    os.makedirs(config.save_path + f"/model_{wandb_run.name}_META", exist_ok=True)
    torch.save(
        {
            "gen_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "it": it,
            "epoch": epoch,
        },
        config.save_path
        + f"/model_{wandb_run.name}_META/it_{it}_loss_{str(loss.item())[:6]}.pt",
    )

    torch.save(
        {
            "gen_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "it": it,
            "epoch": epoch,
        },
        config.save_path + f"/model_{wandb_run.name}_META/latest.pt",
    )
