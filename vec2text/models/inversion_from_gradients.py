import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float
from torch import Tensor as T
from torch.func import functional_call, grad, vmap

from vec2text.models.config import InversionConfig, NEW_ATTRIBUTES
from vec2text.models.inversion import InversionModel


def compute_loss(model):
    """
    Compute the loss for a given model. Needed for ultimately getting the gradients of the model.
    """
    loss_fn = nn.CrossEntropyLoss()

    def compute_loss_(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        model_output = functional_call(
            model,
            (params, buffers),
            (batch, targets),
        )
        return loss_fn(
            model_output.logits.view(-1, model_output.logits.shape[-1]),
            targets.view(-1),
        )

    return compute_loss_

class InversionFromGradientsModel(InversionModel):
    """
    Inversion from gradients model.
    """
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedder_is_decoder = True

        self.embedder_params = {k: v.detach().to("cuda") for k, v in self.embedder.named_parameters()}
        self.embedder_buffers = {k: v.detach().to("cuda") for k, v in self.embedder.named_buffers()}
        self.reduction_version_SVD = getattr(self.config, NEW_ATTRIBUTES["reduction_version_SVD"], None)
        self.reduction_version_JL = getattr(self.config, NEW_ATTRIBUTES["reduction_version_JL"], None)
        # Initialize gradients from config
        self.input_data_gradients = []

        for attr in NEW_ATTRIBUTES:
            if "gradient" in attr:
                value = getattr(self.config, attr, None)
                if value is True:
                    self.input_data_gradients.append(NEW_ATTRIBUTES[attr])


    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs_str = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        emb_input_ids = self.embedder_tokenizer(
            inputs_str,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.encoder_decoder.device)

        # pad input_ids to match the shape of emb_input_ids
        input_ids = nn.functional.pad(input_ids, (0, emb_input_ids.input_ids.shape[1] - input_ids.shape[1]))
        
        ft_compute_grad = grad(compute_loss(self.embedder))
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        ft_per_sample_grads = ft_compute_sample_grad(
            self.embedder_params,
            self.embedder_buffers,
            emb_input_ids.input_ids,
            input_ids,
        )

        return ft_per_sample_grads

    def embed_and_project(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", attention_mask.shape)
        
        grads = self.call_embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        Vs = []
        # TODO: make sure to get specific gradients indicated by the specification.
        for k, g in grads.items():
            if any(elem in k for elem in self.input_data_gradients):
                print(f"k: {k}", flush=True)
        # g: [batch, d1, d2] or [batch, d]
                if g.ndim == 2:
                    g = g.unsqueeze(1)  # [batch, 1, d]
                # Now g: [batch, d1, d2]
                batch_vs = []
                for b in range(g.shape[0]):
                    g_b = g[b]  # [d1, d2]
                    # SVD: get V (right singular vectors)
                    # torch.svd_lowrank returns U, S, V; V: [d2, q]
                    _, _, V = torch.svd_lowrank(g_b, q=1)
                    batch_vs.append(V.squeeze(-1))  # [d2]
                V = torch.stack(batch_vs, dim=0)  # [batch, d2]
                Vs.append(V)

        # Concatenate all Vs along last dim: [batch, total_dim]
        reduced_grad = torch.cat(Vs, dim=-1)

        # Pad to be divisible by encoder_hidden_dim (T5's d_model)
        hidden_dim = self.encoder_hidden_dim
        total_dim = reduced_grad.shape[-1]
        remainder = total_dim % hidden_dim
        if remainder != 0:
            num_zeros_to_add = hidden_dim - remainder
            reduced_grad = nn.functional.pad(reduced_grad, (0, num_zeros_to_add))
        else:
            num_zeros_to_add = 0

        # Reshape to [batch, seq_len, hidden_dim]
        seq_len = reduced_grad.shape[-1] // hidden_dim
        reduced_grad = reduced_grad.view(reduced_grad.shape[0], seq_len, hidden_dim)

        # Attention mask: [batch, seq_len]
        attention_mask = torch.ones((reduced_grad.shape[0], seq_len), device=reduced_grad.device)

        assert reduced_grad.shape == (
            attention_mask.shape[0],
            attention_mask.shape[1],
            hidden_dim,
        )
        return reduced_grad, attention_mask


    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        inputs_embeds, attention_mask = self.embed_and_project(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                # decoder_attention_mask=inputs["decoder_attention_mask"],
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                **generation_kwargs,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        inputs_embeds, attention_mask = self.embed_and_project(
            input_ids=input_ids,
            attention_mask=attention_mask,
            frozen_embeddings=frozen_embeddings,
        )

        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
        )