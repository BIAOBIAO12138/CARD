import torch
from torch import nn
from torch.nn import functional as F

from rqvae4.models.layers import MLPLayers
from rqvae4.models.rq import ResidualVectorQuantizer


class NURQVAE(nn.Module):
    def __init__(
        self,
        in_dim=768,
        num_emb_list=None,
        e_dim=64,
        layers=None,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=100,
        sk_epsilons=None,
        sk_iters=100,
        nvq_hidden_dim=None,
        nvq_loss_weight=1.0,
        nvq_nonlinearity="kumaraswamy",
    ):
        super().__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.nvq_hidden_dim = nvq_hidden_dim or e_dim
        self.nvq_loss_weight = nvq_loss_weight
        self.nvq_nonlinearity = nvq_nonlinearity

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn,
        )
        self.a_raw = nn.Parameter(torch.zeros(self.e_dim))
        self.b_raw = nn.Parameter(torch.zeros(self.e_dim))
        self.alpha_raw = nn.Parameter(torch.zeros(self.e_dim))
        self.x0_raw = nn.Parameter(torch.zeros(self.e_dim))

        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            sk_epsilons=self.sk_epsilons,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_iters=self.sk_iters,
        )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn,
        )

    def _get_kuma_params(self):
        eps = 1e-6
        a = F.softplus(self.a_raw) + eps
        b = F.softplus(self.b_raw) + eps
        return a, b

    def _get_logistic_params(self, z):
        eps = 1e-6
        reduce_dims = tuple(range(z.dim() - 1))
        x_min = z.amin(dim=reduce_dims)
        x_max = z.amax(dim=reduce_dims)
        delta = (x_max - x_min).clamp_min(eps)

        alpha_base = F.softplus(self.alpha_raw) + eps
        alpha = alpha_base / delta

        x0 = torch.sigmoid(self.x0_raw) * delta + x_min
        return alpha, x0, x_min, x_max

    def _logistic_h(self, z):
        eps = 1e-6
        alpha, x0, x_min, x_max = self._get_logistic_params(z)

        bias = torch.sigmoid(alpha * (x_min - x0))
        top = torch.sigmoid(alpha * (x_max - x0))
        scale = (top - bias).clamp_min(eps)

        y = torch.sigmoid(alpha * (z - x0))
        y = (y - bias) / scale
        return y.clamp(eps, 1.0 - eps)

    def _logistic_h_inv(self, y):
        eps = 1e-6
        y = y.clamp(eps, 1.0 - eps)

        alpha, x0, x_min, x_max = self._get_logistic_params(y)

        bias = torch.sigmoid(alpha * (x_min - x0))
        top = torch.sigmoid(alpha * (x_max - x0))
        scale = (top - bias).clamp_min(eps)

        y_scaled = (scale * y + bias).clamp(eps, 1.0 - eps)
        z = torch.log(y_scaled / (1.0 - y_scaled)) / alpha + x0
        return z

    def _kuma_h(self, z):
        eps = 1e-6
        x = torch.sigmoid(z).clamp(eps, 1.0 - eps)
        a, b = self._get_kuma_params()
        x_pow_a = x.pow(a)
        inner = (1.0 - x_pow_a).clamp(eps, 1.0)
        y = 1.0 - inner.pow(b)
        return y.clamp(eps, 1.0 - eps)

    def _kuma_h_inv(self, y):
        eps = 1e-6
        y = y.clamp(eps, 1.0 - eps)
        a, b = self._get_kuma_params()
        one_minus_y = (1.0 - y).clamp(eps, 1.0)
        inner = 1.0 - one_minus_y.pow(1.0 / b)
        inner = inner.clamp(eps, 1.0 - eps)
        x = inner.pow(1.0 / a)
        x = x.clamp(eps, 1.0 - eps)
        z = torch.log(x / (1.0 - x))
        return z

    def _nvq_h(self, z):
        if self.nvq_nonlinearity == "kumaraswamy":
            return self._kuma_h(z)
        if self.nvq_nonlinearity == "logistic":
            return self._logistic_h(z)
        raise ValueError("Unknown nvq_nonlinearity")

    def _nvq_h_inv(self, y):
        if self.nvq_nonlinearity == "kumaraswamy":
            return self._kuma_h_inv(y)
        if self.nvq_nonlinearity == "logistic":
            return self._logistic_h_inv(y)
        raise ValueError("Unknown nvq_nonlinearity")

    def forward(self, x, use_sk=True, return_latent=False):
        z = self.encoder(x)
        z_prime = self._nvq_h(z)
        zq_prime, rq_loss, indices = self.rq(z_prime, use_sk=use_sk)
        z_q = self._nvq_h_inv(zq_prime)
        z_recon_via_nvq = self._nvq_h_inv(z_prime)
        nvq_loss = F.mse_loss(z_recon_via_nvq, z, reduction="mean")

        total_quant_loss = rq_loss + self.nvq_loss_weight * nvq_loss

        out = self.decoder(z_q)

        if return_latent:
            return out, total_quant_loss, indices, {
                "z": z.detach(),
                "z_prime": z_prime.detach(),
                "z_q": z_q.detach(),
            }
        return out, total_quant_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        z = self.encoder(xs)
        z_prime = self._nvq_h(z)
        _, _, indices = self.rq(z_prime, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):
        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError("incompatible loss type")

        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return loss_total, loss_recon
