import torch
from torch import nn

class MultiTaskLoss(nn.Module):
    """ Wrapper class for multi-task learning.
        This class uses the technique from the Multi-Task Learning paper by Kendall et al [arXiv:1705.07115v3] cf. sec. 3.2
    """

    def __init__(self, weights_dict, use_mtl_uncertainty=False):
        """
        Arguments:
            weight_dict: dictionary containing {task_name: init_weights}
        """
        super().__init__()
        self.weights_dict = weights_dict # relative weight of each task loss
        n_tasks = len(weights_dict)

        if use_mtl_uncertainty:
            self.log_sigmas = nn.ParameterDict({key: torch.nn.Parameter(torch.zeros(1)) for key in weights_dict.keys()}) # uncertainty of each task loss
        self.use_mtl_uncertainty = use_mtl_uncertainty

    def forward(self, loss_dict):
        """
        Arguments:
            loss_dict: dictionary containing the loss value calculated for each task. Each item should support back propagation.

        Return:
            final_loss: for back propagation
        """

        final_loss = 0
        assert len(loss_dict) == len(self.weights_dict), f"loss_dict should have the same number of items as weights_dict!"
        for task, loss in loss_dict.items():
            assert task in self.weights_dict, f"Got unexpected task: {task}!"
            if self.use_mtl_uncertainty:
                final_loss += loss * torch.exp(-self.log_sigmas[task]) * self.weights_dict[task] + self.log_sigmas[task]
            else:
                final_loss += loss * self.weights_dict[task]

        loss_dict = {task: loss.item() for task, loss in loss_dict.items()}

        return {'total': final_loss, **loss_dict}

    def get_uncertainty(self):
        """
        return dictionary of uncertainty value for each task loss for logging
        """
        if not self.use_mtl_uncertainty:
            return {}

        return {f"{task}": torch.exp(log_sigma).item() for task, log_sigma in self.log_sigmas.items()}
