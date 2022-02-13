import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKCrossEntropyLoss(nn.Module):
    """ Calculate the CrossEntropyLoss only for the top-k procent hardest pixels in each minibatch
        also known as hard-pixel-mining
        cf. http://arxiv.org/abs/1605.06885
        The original author used top_k = 512
    """

    def __init__(self, top_k, ignore_index=-100, weight=None):
        super().__init__()
        assert 0 < top_k <= 1, f"Got invalid argument top_k={top_k}. Should be in range (0,1]."

        self.top_k = top_k
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, input, target):
        """ input: shape [b x C x H x W], logits
            target: shape [b x H x W], gt labels
        """
        loss_pixel = self.criterion(input, target).contiguous().view(-1)

        if self.top_k == 1:
            return loss_pixel.mean()

        top_k_pixels = int(self.top_k * loss_pixel.numel())
        top_k_loss, _ = torch.topk(loss_pixel, top_k_pixels)

        return top_k_loss.mean()


if __name__ == '__main__':
    input = torch.rand((3,4,5,6))
    target = torch.zeros((3,5,6), dtype=torch.long)
    criterion = TopKCrossEntropyLoss(1)
    loss = criterion(input, target)
    print(loss)
