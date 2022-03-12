import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class QuantizeEMA(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embeddings = torch.rand(dim, n_embed).mul_(2 * math.sqrt(3.0)).sub_(math.sqrt(3.0))
        self.register_buffer("embeddings", embeddings)
        self.register_buffer("ema_cluster_size_hidden", torch.zeros(n_embed))
        self.register_buffer("ema_dw_hidden", torch.zeros_like(embeddings))
        self.register_buffer('counter', torch.tensor(0, dtype=torch.int64))
        self.dist_world_size = torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1

    @property
    def ema_cluster_size(self):
        return self.ema_cluster_size_hidden / (1. - torch.pow(self.decay, self.counter))

    @property
    def ema_dw(self):
        return self.ema_dw_hidden / (1. - torch.pow(self.decay, self.counter))

    def forward(self, input):
        # flatten = input.reshape(-1, self.dim)
        x = input.permute(0, 2, 3, 1)
        flatten = x.reshape(-1, x.size(-1))
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embeddings
            + self.embeddings.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            if self.dist_world_size > 1:
                torch.distributed.all_reduce(embed_onehot_sum)
                torch.distributed.all_reduce(embed_sum)

            # Update EMA
            self.ema_cluster_size_hidden.data.add_(embed_onehot_sum - self.ema_cluster_size_hidden.data, alpha=1 - self.decay)
            self.ema_dw_hidden.data.add_(embed_sum - self.ema_dw_hidden.data, alpha=1 - self.decay)
            self.counter.data.add_(1)

            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.ema_dw / cluster_size.unsqueeze(0)
            self.embeddings.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        codes = F.embedding(embed_id, self.embeddings.transpose(0, 1))
        return codes.permute(0, 3, 1, 2).contiguous()


class Quantize(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, dim, n_embed, beta=0.25):
        super().__init__()
        self.n_e = n_embed
        self.e_dim = dim
        self.beta = beta

        rand_range = 1.0 / self.n_e  # math.sqrt(3)
        self.embeddings = nn.Parameter(torch.rand(dim, n_embed).mul_(2 * rand_range).sub_(rand_range))

    def forward(self, input):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        x = input.permute(0, 2, 3, 1)
        flatten = x.reshape(-1, x.size(-1))
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embeddings
            + self.embeddings.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        # compute loss for embedding
        loss = torch.mean((quantize.detach() - input).pow(2)) + self.beta * \
            torch.mean((quantize - input.detach()).pow(2))

        # preserve gradients
        quantize = input + (quantize - input).detach()
        return quantize, loss, embed_ind

    def embed_code(self, embed_id):
        codes = F.embedding(embed_id, self.embeddings.transpose(0, 1))
        return codes.permute(0, 3, 1, 2).contiguous()


class QuantizeOld(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, e_dim, n_e, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # .........\end

        # with:
        # .........\start
        # min_encoding_indices = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        min_encoding_indices = min_encoding_indices.view(z.shape[:-1])
        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
