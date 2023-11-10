import torch
from torch import nn

class XentBoeck(torch.nn.Module):
    '''Boeck cross-entropy loss'''
    def __init__(self, boeck_window = torch.Tensor([[[0.1, 0.2, 0.4, 0.2, 0.1]]]), reduction='mean', device='cpu'):
        super(XentBoeck, self).__init__()
        self.device = device
        self.xent = nn.CrossEntropyLoss(reduction=reduction)
        # self.xent = nn.NLLLoss(reduction=reduction)
        self.boeck_window = boeck_window
        assert torch.sum(self.boeck_window).item() == 1.0, 'boeck window should sum to one, but got {}'.format(self.boeck_window)
        assert self.boeck_window.shape[2]%2 != 0, 'Boeck window should be of odd length, but got {}'.format(len(self.boeck_window))

    def generate_boeck_target(self, preds: torch.Tensor,  labels_cls_idx: torch.Tensor) -> torch.Tensor:
        '''
        Generate Boeck target by convolving one-hot vectors with the boeck window
        labels_cls_idx of shape (batch_size)'''
        conv = nn.Conv1d(1, 1, self.boeck_window.shape[2], bias=False, padding='same')
        conv.weight = torch.nn.Parameter(self.boeck_window)
        one_hot = torch.zeros(labels_cls_idx.shape[-1], 1, preds.shape[-1]) #include channel dimension for conv
        for i, idx in enumerate(labels_cls_idx):
            one_hot[i, 0, int(idx.item())] = 1.0
        return torch.squeeze(conv(one_hot)) #remove channel dimension to output shape (batch_size, n_classes)

    def forward(self, preds, labels_cls_idx):
        boeck_target = self.generate_boeck_target(preds, labels_cls_idx).to(self.device)
        loss = self.xent(preds, boeck_target)
        return loss