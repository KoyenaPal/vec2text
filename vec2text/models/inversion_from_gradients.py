import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float
from torch import Tensor as T
from torch.func import functional_call, grad, vmap
from torch.utils._pytree import tree_map
from vec2text.models.config import InversionConfig, NEW_ATTRIBUTES
from vec2text.models.inversion import InversionModel
from torch.autograd import grad as autograd_grad

def compute_loss(model):
    """
    Compute the loss for a given model. Needed for ultimately getting the gradients of the model.
    """
    loss_fn = nn.CrossEntropyLoss()

    def compute_loss_(params, buffers, sample, target):
        print("sample shape:", sample.shape)
        print("target shape:", target.shape)
        print("params:", params)
        print("buffers:", buffers)
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        # For GPT models, the forward signature is typically (input_ids, attention_mask, labels)
        # where labels are used for language modeling loss
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
        # self.reduction_version_SVD = getattr(self.config, NEW_ATTRIBUTES["SVD"], None)
        # self.reduction_version_JL = getattr(self.config, NEW_ATTRIBUTES["JL"], None)
        # Initialize gradients from config
        self.input_data_gradients = []

        for attr in NEW_ATTRIBUTES:
            if "gradient" in attr:
                value = getattr(self.config, attr, None)
                if value is True:
                    self.input_data_gradients.append(NEW_ATTRIBUTES[attr])
        #self.per_batch = getattr(self.config, NEW_ATTRIBUTES["per_batch"], None)


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
            # print("embedder buffers shape:", self.embedder_buffers.shape)
            # print("embedder params shape:", self.embedder_params.shape)
            # print("emb_input_ids shape:", emb_input_ids.input_ids.shape)
            # print("input_ids shape:", input_ids.shape)
            ft_per_sample_grads = ft_compute_sample_grad(
                self.embedder_params,
                self.embedder_buffers,
                emb_input_ids.input_ids,
                input_ids,
            )
            print("ft_per_sample_grads:", ft_per_sample_grads.keys(), flush=True)
            if not self.config.per_batch:
                print("Came to per sample", flush=True)
                return ft_per_sample_grads
            else:
                print("Came to FULL GRAD", flush=True)
                return {k: v.sum(dim=0, keepdim=True) for k, v in ft_per_sample_grads.items()}
                #return ft_per_sample_grads.sum(dim=0, keepdim=True)
            # return ft_per_sample_grads
    
    def embed_and_project_svd(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grads = self.call_embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        Vs = []
        # Calculate target SVD rank to ensure consistent dimensions
        # We want the final concatenated dimension to be divisible by encoder_hidden_dim
        target_total_dim = self.encoder_hidden_dim * 4  # Use 4 as a reasonable sequence length
        num_gradients = sum(1 for k, g in grads.items() 
                            if any(elem in k for elem in self.input_data_gradients))
        
        if num_gradients > 0:
            # Distribute the target dimension evenly across gradients
            target_svd_rank = max(1, target_total_dim // num_gradients)
            print(f"target_svd_rank: {target_svd_rank}", flush=True)
            # Ensure the rank doesn't exceed the minimum dimension of any gradient
            min_grad_dim = float('inf')
            for k, g in grads.items():
                if any(elem in k for elem in self.input_data_gradients):
                    if g.ndim == 2:
                        min_grad_dim = min(min_grad_dim, g.shape[-1])
                    else:
                        min_grad_dim = min(min_grad_dim, g.shape[-2], g.shape[-1])
            
            target_svd_rank = min(target_svd_rank, min_grad_dim)
            #target_svd_rank = 1
            print(f"Using consistent SVD rank: {target_svd_rank} for {num_gradients} gradients")
            
            for k, g in grads.items():
                if any(elem in k for elem in self.input_data_gradients):
                    print(f"k: {k}", flush=True)
                    # g: [batch, d1, d2] or [batch, d]
                    print(f"g shape: {g.shape}", flush=True)
                    if g.ndim == 2:
                        g = g.unsqueeze(1)  # [batch, 1, d]
                    # Now g: [batch, d1, d2]
                    batch_vs = []
                    for b in range(g.shape[0]):
                        g_b = g[b]  # [d1, d2]
                        # SVD: get V (right singular vectors) with consistent rank
                        # torch.svd_lowrank returns U, S, V; V: [d2, target_svd_rank]
                        _, _, V = torch.svd_lowrank(g_b, q=target_svd_rank)
                        batch_vs.append(V)  # [d2, target_svd_rank]
                    V = torch.stack(batch_vs, dim=0)  # [batch, d2, target_svd_rank]
                    print(f"V shape: {V.shape}", flush=True)
                    # Flatten the last two dimensions: [batch, d2 * target_svd_rank]
                    V = V.view(V.shape[0], -1)
                    print(f"V FLATTEN LAST TWO DIMENSIONS shape: {V.shape}", flush=True)
                    assert len(V.shape) == 2, f"V shape: {V.shape}"
                    Vs.append(V)

        # # Concatenate all Vs along last dim: [batch, total_dim]
        # reduced_grad = torch.cat(Vs, dim=-1)

        # # Pad to be divisible by encoder_hidden_dim (T5's d_model)
        # hidden_dim = self.encoder_hidden_dim
        # total_dim = reduced_grad.shape[-1]
        # remainder = total_dim % hidden_dim
        # if remainder != 0:
        #     num_zeros_to_add = hidden_dim - remainder
        #     reduced_grad = nn.functional.pad(reduced_grad, (0, num_zeros_to_add))
        # else:
        #     num_zeros_to_add = 0

        # # Reshape to [batch, seq_len, hidden_dim]
        # seq_len = reduced_grad.shape[-1] // hidden_dim
        # reduced_grad = reduced_grad.view(reduced_grad.shape[0], seq_len, hidden_dim)

        # # Attention mask: [batch, seq_len]
        # attention_mask = torch.ones((reduced_grad.shape[0], seq_len), device=reduced_grad.device)

        # assert reduced_grad.shape == (
        #     attention_mask.shape[0],
        #     attention_mask.shape[1],
        #     hidden_dim,
        # )
        return Vs

    def embed_and_project_jl(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grads = self.call_embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        Vs = []
        # maybe increase the target to 8? initial target is 4
        target_total_dim = self.encoder_hidden_dim * 8
        print(f"target_total_dim: {target_total_dim}", flush=True)
        num_gradients = sum(1 for k, g in grads.items() 
                            if any(elem in k for elem in self.input_data_gradients))
        
        if num_gradients > 0:
            # Distribute the target dimension evenly across gradients
            target_jl_dim = max(1, target_total_dim // num_gradients)
            print(f"Using JL reduction to dimension: {target_jl_dim} for {num_gradients} gradients")
            
            for k, g in grads.items():
                if any(elem in k for elem in self.input_data_gradients):
                    print(f"k: {k}", flush=True)
                    # g: [batch, d1, d2] or [batch, d]
                    if g.ndim == 2:
                        g = g.unsqueeze(1)  # [batch, 1, d]
                    # Now g: [batch, d1, d2]
                    batch_reduced = []
                    for b in range(g.shape[0]):
                        g_b = g[b]  # [d1, d2]
                        # Flatten to 1D: [d1 * d2]
                        g_flat = g_b.view(-1)
                        
                        # Create random projection matrix for JL
                        input_dim = g_flat.shape[0]
                        if not hasattr(self, f'jl_projection_{k}'):
                            # Initialize random projection matrix (Gaussian)
                            projection = torch.randn(target_jl_dim, input_dim, device=g_flat.device)
                            # Normalize to preserve distances approximately
                            projection = projection / torch.sqrt(torch.tensor(target_jl_dim, device=g_flat.device))
                            setattr(self, f'jl_projection_{k}', projection)
                        
                        projection = getattr(self, f'jl_projection_{k}')
                        # Apply JL projection: [target_jl_dim] x [input_dim] = [target_jl_dim]
                        reduced_g_b = torch.matmul(projection, g_flat)
                        batch_reduced.append(reduced_g_b)
                    
                    V = torch.stack(batch_reduced, dim=0)  # [batch, target_jl_dim]
                    Vs.append(V)
        return Vs

    def embed_and_project(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", attention_mask.shape)
        
        Vs = []
        print("self.config.reduction_version:", self.config.reduction_version, flush=True)
        if self.config.reduction_version == "SVD":
            Vs = self.embed_and_project_svd(input_ids, attention_mask)
        elif self.config.reduction_version == "JL":
            Vs = self.embed_and_project_jl(input_ids, attention_mask)
        else:
            raise ValueError(f"Invalid reduction version: {self.config.reduction_version}")
            
        
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
        
    def embed_and_project_param_diff(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        frozen_embeddings: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Side-effect free: NO backward(), NO optimizer.step().
        Computes a hypothetical per-batch update Δθ via functional grads and projects it.
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        grads = self.call_embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get parameters that need gradients
        params_dict = dict(self.embedder.named_parameters())
        buffers_dict = dict(self.embedder.named_buffers())
        
        tracked_params = [(n, p) for n, p in params_dict.items() 
                         if p.requires_grad and any(tag in n for tag in self.input_data_gradients)]
        print(tracked_params, flush=True)
        if not tracked_params:
            # Return zero embedding if no parameters to track
            hidden_dim = self.encoder_hidden_dim
            zeros = torch.zeros(batch_size, 1, hidden_dim, device=device)
            mask = torch.ones(batch_size, 1, device=device)
            return zeros, mask

        # Compute loss and gradients
        # compute_loss_fn = compute_loss(self.embedder)
        # loss = compute_loss_fn(params_dict, buffers_dict, input_ids, input_ids)
        
        # Get gradients for tracked parameters only
        param_tensors = [p for _, p in tracked_params]
        # grads = autograd_grad(loss, param_tensors, create_graph=False, retain_graph=False, allow_unused=False)

        # Get optimizer hyperparameters
        pg = self.optimizer.param_groups[0] if self.optimizer.param_groups else {}
        lr = pg.get("lr", 1e-3)
        weight_decay = pg.get("weight_decay", 0.0)

        # Compute parameter updates
        deltas = {}
        with torch.no_grad():
            for (name, param), grad in zip(tracked_params, grads):
                if weight_decay != 0.0:
                    grad = grad + weight_decay * param
                deltas[name] = -lr * grad

        # JL projection
        target_total_dim = self.encoder_hidden_dim * 8
        target_jl_dim = max(1, target_total_dim // len(deltas))
        
        Vs = []
        for name, delta in deltas.items():
            # Expand delta to batch dimension
            delta_batched = delta.unsqueeze(0).expand(batch_size, *delta.shape)
            if delta_batched.ndim == 2:
                delta_batched = delta_batched.unsqueeze(1)
            
            # Flatten and project
            flat_deltas = delta_batched.view(batch_size, -1)  # [B, param_size]
            input_dim = flat_deltas.size(-1)
            
            # Get or create projection matrix
            proj_attr = f'jl_projection_{name}'
            if not hasattr(self, proj_attr):
                proj = torch.randn(target_jl_dim, input_dim, device=device)
                proj = proj / torch.sqrt(torch.tensor(target_jl_dim, device=device))
                setattr(self, proj_attr, proj)
            
            proj = getattr(self, proj_attr)
            reduced = torch.matmul(flat_deltas, proj.T)  # [B, target_jl_dim]
            Vs.append(reduced)

        # Concatenate and reshape
        reduced_update = torch.cat(Vs, dim=-1)  # [B, total_dim]
        hidden_dim = self.encoder_hidden_dim
        
        # Pad to be divisible by hidden_dim
        total_dim = reduced_update.size(-1)
        if total_dim % hidden_dim != 0:
            pad_size = hidden_dim - (total_dim % hidden_dim)
            reduced_update = nn.functional.pad(reduced_update, (0, pad_size))
        
        # Reshape to [B, T, H]
        seq_len = reduced_update.size(-1) // hidden_dim
        reduced_update = reduced_update.view(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        return reduced_update, attention_mask





    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        
        # Debug: print what's in the inputs
        # print("DEBUG: inputs keys:", list(inputs.keys()))
        # print("DEBUG: input_ids in inputs:", "input_ids" in inputs)
        # print("DEBUG: attention_mask in inputs:", "attention_mask" in inputs)
        # print("DEBUG: frozen_embeddings in inputs:", "frozen_embeddings" in inputs)
        
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