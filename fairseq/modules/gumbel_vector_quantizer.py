# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim:int,
        num_vars:int,
        temp,
        groups:int,
        combine_groups,
        vq_dim:int,
        time_first:bool,
        activation=nn.GELU(),
        weight_proj_depth:int=1,
        weight_proj_factor=1,
    ):
        """Vector quantization using gumbel softmax

        Multiple methodologies are supported.
            - Product Quantization (Grouping)
            - Temperature decay

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group (size of sub-codebook; `V`)
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization; `G`
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: input:(Batch, Time, Channel) if True else (Batch, Channel, Time)
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        # Dimension of each group's representative vector
        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        # Full Codebook: (1, G*V, var_dim)
        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        nn.init.uniform_(self.vars)

        # Projection: Linear/non-Linear projection for dimension matching
        #     depth>1: vec_i::(B, T, C) -> [FC_{factor}-σ]x{depth} -> (B, T, G*V)
        #     depth=1: vec_i::(B, T, C) -> Linear -> (B, T, G*V)
        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                """FC-σ"""
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)
        # /

        # Gumbel-softmax Temperature
        if isinstance(temp, str):
            import ast
            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"
        self.max_temp, self.min_temp, self.temp_decay = temp
        ## Current temperature, initialized with maximum
        self.curr_temp = self.max_temp
        # /

        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        """(maybe) Cool sampler.

        Lower the temperature of sampler.
        This procedure is idempotent (Call twice do NOT cool two times, there is no cooling history)
        Args:
            num_updates: Number of cool/decay
        """
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_vars ** self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            # (1, G*V, var_dim) =>  (G*V, var_dim)
            self.vars.squeeze(0)
            # (G*V, var_dim) => (dim_indices, var_dim)
            .index_select(0, indices)
            .view(self.num_vars ** self.groups, -1)
        )

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert (
            n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.num_vars ** exponent)
        return res

    def forward_idx(self, x):
        """Convert feature sequences into quantized representative and index series.

        Args:
            x: Input (Batch, Time, Feature) if time_first else (Batch, Feature, Time)
        Returns:
            Sampled quantized series - (representative:(B, T, vq_dim), index:(B, T, G))
        """
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):
        """
        Quantize input feature sequences.

        Args:
            x: Input (Batch, Time, Feature) if time_first else (Batch, Feature, Time)
        Returns:
            Dict {
            'num_vars': Number of sub-codes (V*G)
            'code_perplexity'
            'prob_perplexity'
            'temp': Temperature at sampling
            'x' (B, T, vq_dim==G*var_dim): Sampled representative vector series
            'targets' Optional[(B, T, G)]: Index vector series for training targets
        }
        """

        # Output dictionary
        result = {"num_vars": self.num_vars * self.groups}

        # ((Batch, Time, Feature) | (Batch, Feature, Time)) => (B, T, F)
        if not self.time_first:
            x = x.transpose(1, 2)

        # B, T, F
        bsz, tsz, fsz = x.shape
        # Reshape because quantization is done toward each vector: (B, T, F) => (B*T, F)
        x = x.reshape(-1, fsz)
        # Feature projection: (B*T, F) => (B*T, G*V)
        x = self.weight_proj(x)
        # Grouping: (B*T, G*V) => (B*T*G, V)
        x = x.view(bsz * tsz * self.groups, -1)

        # Undifferentiable sampling: Argmax
        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()
        # /

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        # Output 4: Temperature
        result["temp"] = self.curr_temp

        # Differential hard random sampling
        if self.training:
            # (B*T*G, V) => (B*T*G, V), last dim is one-hot vector
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        # Differential is not necessary, so Argmax sampling is enough.
        else:
            x = hard_x

        # degrouping/concat: (B*T*G, V) => (B*T, G*V)
        x = x.view(bsz * tsz, -1)

        # (1, G*V, var_dim)
        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        # Produce index series for training target
        if produce_targets:
            result["targets"] = (
                # (B*T, G*V) => (B*T*G, V)
                x.view(bsz * tsz * self.groups, -1)
                # (B*T*G, V) => (B*T*G), values are index
                .argmax(dim=-1)
                # (B*T*G) => (B, T, G)
                .view(bsz, tsz, self.groups)
                .detach()
            )

        # one-hot vectors to representative vectors: (B*T, G*V, 1) * (1, G*V, var_dim) => (B*T, G*V, var_dim)
        # Most row is zero vector, hot rows are converted to the vectors, so G non-zero rows & G(V-1) zero rows
        x = x.unsqueeze(-1) * vars
        # Squash zero raws in each groups, then become concatenated representative vectors
        #     (B*T, G*V, var_dim) => (B*T, G, V, var_dim) => (B*T, G, var_dim) => (B, T, G*var_dim == vq_dim)
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1).sum(-2).view(bsz, tsz, -1)

        # (B, T, vq_dim) => ((B, T, vq_dim) | (B, vq_dim, T)) same as input
        if not self.time_first:
            x = x.transpose(1, 2)

        result["x"] = x

        return result
