import torch.nn.functional as F
import torch
import torch.nn as nn


class AttLayer2(nn.Module):
    """Soft alignment attention implement.

    Attributes:
        dim (int): attention hidden dim
    """

    def __init__(self, dim=768, seed=0):
        """Initialization steps for AttLayer2.

        Args:
            dim (int): attention hidden dim
        """
        super(AttLayer2, self).__init__()
        self.dim = dim
        self.seed = seed
        torch.manual_seed(seed)

        self.W = None
        self.b = None

        self.q = nn.Parameter(torch.Tensor(dim, 1))
        nn.init.xavier_uniform_(self.q)

    def forward(self, inputs, mask=None):
        """Core implementation of soft attention

        Args:
            inputs (Tensor): input tensor of shape (B, seq_len, input_dim).
            mask (Tensor, optional): mask tensor. Defaults to None.

        Returns:
            Tensor: weighted sum of input tensors (B, input_dim).
        """
        input_dim = inputs.size(-1)
        
        if self.W is None or self.b is None or self.W.size(0) != input_dim:
            self.W = nn.Parameter(torch.empty(input_dim, self.dim).to(inputs.device))
            self.b = nn.Parameter(torch.zeros(self.dim).to(inputs.device))
            nn.init.xavier_uniform_(self.W)

        attention = torch.tanh(torch.matmul(inputs, self.W) + self.b)
        attention = torch.matmul(attention, self.q).squeeze(2)

        if mask is None:
            attention = torch.exp(attention)
        else:
            attention = torch.exp(attention) * mask.float()

        attention_weight = attention / (torch.sum(attention, dim=-1, keepdim=True) + 1e-8)
        attention_weight = attention_weight.unsqueeze(2)

        weighted_input = inputs * attention_weight
        return torch.sum(weighted_input, dim=1)

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor

        Args:
            input_shape (tuple): shape of input tensor.

        Returns:
            tuple: shape of output tensor.
        """
        return input_shape[0], input_shape[-1]
