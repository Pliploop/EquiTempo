import os
import time
from tqdm import tqdm
import numpy as np
import torch
from src.model.model import Siamese
from config.train import TrainConfig
from config.dataset import MTATConfig



config = TrainConfig()
dataset_config = MTATConfig()


def init_model(path=None, test=False):
    device = config.device
    model = Siamese(filters=config.filters, dilations=config.dilations, dropout_rate=config.dropout_rate, output_dim=config.output_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
    it = 0
    if path is not None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        it = checkpoint['it']
    if test:
        model.eval()
    return model,optimizer,scaler,it


def save_model(loss, it, model, optimizer, scaler):
    os.makedirs(config.save_path, exist_ok=True)
    torch.save({
            'gen_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': loss,
            'it': it,
            }, config.save_path+f'/model_loss_{str(loss)[:6]}_it_{it}.pt')


def loss_function(c1, c2, alpha1, alpha2, eps=1e-7):
    c_ratio = c1/(c2+eps)
    alpha_ratio = alpha1/(alpha2+eps)
    return torch.abs(c_ratio-alpha_ratio)


def train_iteration(x1, x2, alpha1, alpha2, model, optimizer, scaler):
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.mixed_precision):
        _,c1 = model(x1)
        _,c2 = model(x2)
        loss = loss_function(c1,c2,alpha1,alpha2)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def update_lr(new_lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train_loop(dataloader, model, optimizer, scaler, it=0, writer=None):
    dataloader_length = len(dataloader)
    if config.warmup:
        update_lr(1e-7, optimizer)
        target_lr = config.lr
        lr = 1e-7
    else:
        update_lr(config.lr, optimizer)
    model = model.to(config.device)
    model.train()
    # try:
    counter = 0
    loss = 0.
    for epoch in range(config.epochs):
        bef = time.time()
        bef_loop = time.time()
        loss_list = []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.epochs}', leave=True, total=dataloader_length)
        for batch_i,data in enumerate(pbar):
            loss = train_iteration(data['audio_1'].to(config.device),data['audio_2'].to(config.device),data['rp_1'],data['rp_2'], model,optimizer,scaler)
            if writer is not None:
                writer.add_scalar('loss', loss, it)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)
            loss_list.append(loss)
            counter += 1
            it += 1
            if config.warmup:
                if epoch==0:
                    lr = lr + (1/dataloader_length)*target_lr
                    update_lr(lr, optimizer)

            if batch_i%config.display_progress_every==0:
                pbar.set_postfix({'Loss_sc': np.mean(loss_list[-counter:], axis=0),
                                    'Iter': it,
                                    'LR': optimizer.param_groups[0]['lr'],
                                    'Time/Iter': (time.time()-bef_loop)/config.display_progress_every})
                bef_loop = time.time()
        save_model(np.mean(loss_list[-counter:], axis=0), it, model,optimizer,scaler)
        # test_model(model, writer=writer, step=it, device=device)
        counter = 0
    # except Exception as e:
    #     print(e)
    # finally:
    #     save_model(np.mean(loss_list[-counter:], axis=0), it, model,optimizer,scaler)
    #     return it


