import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastivePrototypicalLoss(nn.Module):
    """
    Loss function in CPP
    """
    def __init__(self, temperature=0.6, reduction="mean", cl_negative=0):
        super(ContrastivePrototypicalLoss, self).__init__()
        print('Temperature: ', temperature)
        self.temperature = temperature
        self.reduction = reduction
        self.cl_negative = cl_negative
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, z_feature, label, previous_prototype=None):

        assert z_feature.ndim > 1, "z_feature must have number of dimension > 1."
        batch_size, _ = z_feature.shape
        assert z_feature.shape[0] == label.shape[0], "z_feature.shape[0] != label.shape[0]"

        if z_feature.ndim > 2:
            z_feature = z_feature.reshape(z_feature.shape[0], -1)  # flatten

        z_feature = F.normalize(z_feature, dim=1)

        if previous_prototype is None:
            concat_z_and_prototype = z_feature
            num_prototype = 0
        else:
            assert previous_prototype.ndim > 1, "previous_prototype must have number of dimension > 1."
            if previous_prototype.ndim > 2:
                previous_prototype = previous_prototype.reshape(previous_prototype.shape[0], -1)
            assert z_feature.shape[1] == previous_prototype.shape[1], "z_feature.shape[1] != previous_prototype.shape[1]"
            (num_prototype, _) = previous_prototype.shape
            previous_prototype = F.normalize(previous_prototype, dim=1)
            concat_z_and_prototype = torch.cat([z_feature, previous_prototype], dim=0)

        z_dot_z_T = torch.div(torch.matmul(z_feature, concat_z_and_prototype.T), 
                              self.temperature)

        # create mask_for_same_classes
        mask_for_same_classes = torch.zeros(batch_size, batch_size + num_prototype).to(self._device)
        labels = label.contiguous().view(-1, 1)
        current_task_mask = torch.eq(labels, labels.T).float() # 1 if same class, 0 if not same class
        mask_for_same_classes[:batch_size, :batch_size] = current_task_mask
        mask_for_different_classes = 1 - mask_for_same_classes # 0 if same class, 1 if not same class
        # numerical stability
        max_z_dot_z_T = torch.max(z_dot_z_T, dim=1, keepdim=True).values
        assert max_z_dot_z_T.shape == (batch_size, 1), "max_z_dot_z_T.shape != (batch_size, 1)."
        z_dot_z_T = z_dot_z_T - max_z_dot_z_T.detach()
        num_positive = torch.sum(mask_for_same_classes, dim=1, keepdim=True)
        positive_logits = torch.sum(z_dot_z_T * mask_for_same_classes, dim=1, keepdim=True)
        negative_logits = torch.exp(z_dot_z_T) * mask_for_different_classes
        if self.cl_negative > 0:
            # Take top-k negative logits only
            negative_logits, _ = torch.topk(negative_logits, self.cl_negative, dim=1)
        loss_for_each_instance = (-1 / num_positive).reshape(-1, 1) * positive_logits + \
                                 torch.log(torch.sum(negative_logits, dim=1, keepdim=True))
  
        try:
            assert loss_for_each_instance.shape == (batch_size, 1), "loss_for_each_instance.shape != (batch_size, 1)"
        except AssertionError:
            print(loss_for_each_instance.shape)
            raise RuntimeError("loss_for_each_instance.shape != (batch_size, 1)")

        if self.reduction == "mean":
            return torch.mean(loss_for_each_instance)
        elif self.reduction == "sum":
            return torch.sum(loss_for_each_instance)
        elif self.reduction == "none":
            return loss_for_each_instance.squeeze(-1)

if __name__ == "__main__":
    cpp_loss = ContrastivePrototypicalLoss(temperature=0.6, reduction="none")
    z_feature = torch.Tensor([[0.6011, 1.8857, 0.5336],
                              [-0.8425, -2.2244, 0.6984],
                              [-0.1734, -0.9198, -1.5337],
                              [-0.2995, 0.7109, 1.5230],
                              [-0.2606, 0.1352, -1.2000]]).to(cpp_loss._device)
    label = torch.Tensor([2, 0, 0, 1, 2]).to(cpp_loss._device)
    previous_prototype = torch.Tensor([[ 0.6475, -1.1527, -0.9547],
                                        [-0.0105, -0.9611, -1.0651]]).to(cpp_loss._device)
    loss_value = cpp_loss(z_feature, label)
    print(loss_value[0]) # it should be around -1.4323, according to my manual calculation