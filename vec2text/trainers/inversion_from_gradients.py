from typing import Dict, Optional
import logging
import torch

from vec2text.trainers.inversion import InversionTrainer
import tqdm
import random
import csv
import copy
from typing import List, Tuple
import scipy.stats
import numpy as np
from vec2text.trainers.base import sem, logger


class InversionFromGradientsTrainer(InversionTrainer):
    """Custom trainer for inverting from gradients. Contains special
    decoding methods that we can only use here, mostly that
    have to do with conditioning on a suffix.
    """

    generation_method: Optional[str] = None

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
            return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    
    def _log_preds_table(
        self, table_key: str, decoded_original: List[str], decoded_preds: List[str], decoded_labels: List[str]
    ):
        if not self.args.use_wandb:
            return
        elif not (self.args.local_rank <= 0):
            return
        num_rows = len(decoded_preds) 
        idxs = random.choices(
           range(len(decoded_preds)), k=min(50, num_rows)
        )

        data = []

        for idx in idxs:
            data.append([decoded_original[idx], decoded_labels[idx], decoded_preds[idx]])
        if self.args.use_wandb:
            import wandb
            table = wandb.Table(columns=["Original", "Label", "Decoded"], data=data)
            wandb.log({table_key: table})
        else:
            # Specify the CSV file name
            filename = "eval_output.csv"

            # Write data to the CSV file
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(["Original", "Label", "Decoded"])
                # Write the rows
                writer.writerows(data)

            print(f"Data successfully written to {filename}")

    def _get_decoded_sequences(
        self, dataloader: torch.utils.data.DataLoader, n: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Iterates through eval dataset and does decoding.

        TODO: do this better. We shouldn't need to iterate through eval set twice
        but I don't want to copy 1000 lines of code to change their eval loop...

        Probably want custom eval eventually. Also this depends on eval data being
        in the same order which is annoying.
        """
        assert not self.model.training

        gen_kwargs = copy.copy(self.gen_kwargs)

        all_preds = []
        all_labels = []
        for step, inputs in enumerate(
            tqdm.tqdm(dataloader, desc="generating from val", leave=False)
        ):
            # https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
            inputs_cuda = {k: v.to(self.args.device) for k, v in inputs.items()}
            labels = inputs["labels"]
            max_length = self.model.config.max_seq_length
            gen_kwargs["max_length"] = max_length
            with torch.no_grad():
                generated_text = self.generate(
                    inputs=inputs_cuda, generation_kwargs=gen_kwargs
                )
            if generated_text.shape[1] < max_length:
                # Pad generated text to max length
                pad_tokens = (
                    torch.ones(
                        (generated_text.shape[0], max_length - generated_text.shape[1]),
                        dtype=torch.long,
                        device=generated_text.device,
                    )
                    * self.pad_token_id
                )
                generated_text = torch.cat((generated_text, pad_tokens), dim=1)

            true_input_ids = inputs["input_ids"]
            if true_input_ids.shape[1] < max_length:
                # Pad true text to max length
                # Pad generated text to max length
                pad_tokens = (
                    torch.ones(
                        (true_input_ids.shape[0], max_length - true_input_ids.shape[1]),
                        dtype=torch.long,
                        device=true_input_ids.device,
                    )
                    * self.pad_token_id
                )
                true_input_ids = torch.cat((true_input_ids, pad_tokens), dim=1)

            all_preds.extend(generated_text.cpu().tolist())
            # all_labels.extend((true_input_ids.cpu().tolist(),labels.cpu().tolist()))
            true_input_ids_list = true_input_ids.cpu().tolist()
            labels_list = labels.cpu().tolist()
            batch_pairs = list(zip(true_input_ids_list, labels_list))
            all_labels.extend(batch_pairs)
            if len(all_preds) >= n:
                break

        return all_preds, all_labels
    
    def _clean_token_ids(self, token_ids_list: List[List[int]]) -> List[List[int]]:
        """Replaces negative token IDs with the pad token ID."""
        cleaned_list = []
        pad_token_id = self.tokenizer.pad_token_id
        # Ensure pad_token_id is not None, if it is, maybe raise error or use a default?
        # For now, assume it's valid.
        if pad_token_id is None:
                # Defaulting to 0 if pad_token_id is None, or handle appropriately
                pad_token_id = 0 

        for tokens in token_ids_list:
            cleaned_tokens = [
                token if token >= 0 else pad_token_id for token in tokens
            ]
            cleaned_list.append(cleaned_tokens)
        return cleaned_list

    def eval_generation_metrics(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        # Get decoded text. Note that this is different than `preds`, which
        # is used to compute the loss.
        preds_sample_list, preds_sample_labels_list = self._get_decoded_sequences(
            dataloader=dataloader, n=10000
        )
        print("preds_sample_labels_list", preds_sample_labels_list, flush=True)
        labels_list = []
        if isinstance(preds_sample_labels_list[0], tuple):
            labels_list = [label[1] for label in preds_sample_labels_list]
        else:
            labels_list = preds_sample_labels_list

        # Log BLEU, log table of text.
        cleaned_preds_sample_list = self._clean_token_ids(preds_sample_list)
        decoded_preds = self.tokenizer.batch_decode(
            cleaned_preds_sample_list, skip_special_tokens=True
        )
        decoded_labels = []
        decoded_original = []
        if isinstance(preds_sample_labels_list[0], tuple):
            input_texts = [label[0] for label in preds_sample_labels_list]
            labels = [label[1] for label in preds_sample_labels_list]

            cleaned_labels = self._clean_token_ids(labels)
            decoded_labels = self.tokenizer.batch_decode(
                cleaned_labels, skip_special_tokens=True
            )

            cleaned_input_texts = self._clean_token_ids(input_texts)
            decoded_original = self.tokenizer.batch_decode(
                cleaned_input_texts, skip_special_tokens=True
            )
        else:
            # Assuming preds_sample_labels_list contains the labels in this case
            cleaned_labels = self._clean_token_ids(preds_sample_labels_list)
            decoded_labels = self.tokenizer.batch_decode(
                cleaned_labels, skip_special_tokens=True
            )
            # If original text is needed here, it should also be cleaned
            # For now, assuming decoded_original = decoded_labels is intended
            decoded_original = decoded_labels 

        bleu_result = self._text_comparison_metrics(
            predictions_ids=cleaned_preds_sample_list, # Use cleaned ids
            predictions_str=decoded_preds,
            references_ids=self._clean_token_ids(labels_list), # Use cleaned ids
            references_str=decoded_labels,
        )
        print("function right before log pred table")
        print("length of decoded_preds", len(decoded_preds), flush=True)
        self._log_preds_table(
            table_key="val_text_preds",
            decoded_original=decoded_original,
            decoded_preds=decoded_preds,
            decoded_labels=decoded_labels,
        )

        if not len(decoded_preds):
            return {}
        print("[pred]", decoded_preds[0])
        print("[true]", decoded_labels[0])
        print("\n\n")
        print("[pred]", decoded_preds[1])
        print("[true]", decoded_labels[1])
        print("\n\n")
        print("[pred]", decoded_preds[2])
        print("[true]", decoded_labels[2])

        # Compute sims of eval data using embedder.
        # Need to decide if we use original or cleaned tokens for tensor conversion
        # Using original lists for now, assuming tensor operations handle padding correctly.
        #TODO: 128 is hardcoded here, make it a parameter -- confirm what that should look like
        # update: I would like to remove the 128 limit for now, but we should still make it a parameter
        preds_sample = torch.tensor(preds_sample_list, device=self.args.device)[:128]
        # Handling the tuple case for preds_sample_labels_list tensor conversion
        if isinstance(preds_sample_labels_list[0], tuple):
             # Decide which part of the tuple to use, e.g., the labels part
             labels_for_tensor = [item[1] for item in preds_sample_labels_list]
             preds_sample_labels = torch.tensor(labels_for_tensor, device=self.args.device)[:128]
        else:
             preds_sample_labels = torch.tensor(
                 preds_sample_labels_list, device=self.args.device
             )[:128]

        # Log num tokens.
        # Note: Calculating num tokens based on original token lists before cleaning
        num_tokens_metrics = {
            "pred_num_tokens": (
                (preds_sample != self.pad_token_id)
                & (preds_sample != self.bos_token_id)
            )
            .sum(1)
            .float()
            .mean()
            .item(),
            "true_num_tokens": (
                (preds_sample_labels != self.pad_token_id)
                & (preds_sample_labels != self.bos_token_id)
            )
            .sum(1)
            .float()
            .mean()
            .item(),
        }

        # Fix eos token on generated text.
        # bos_token_id = self.embedder_tokenizer.pad_token_id
        # assert (preds_sample[:, 0] == bos_token_id).all()
        eos_token_id = self.embedder_tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_tokens = (
                torch.ones(
                    (len(preds_sample), 1),
                    dtype=torch.long,
                    device=self.args.device,
                )
                * eos_token_id
            )
            preds_sample = torch.cat((preds_sample[:, 1:], eos_tokens), dim=1)
            # assert preds_sample.shape == preds_sample_labels.shape

        try:
            with torch.no_grad():
                # self.inversion_trainer.model.noise_level = 0.0
                preds_sample_retokenized = self.embedder_tokenizer(
                    decoded_preds,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )["input_ids"].to(preds_sample.device)
                preds_sample_retokenized = preds_sample_retokenized[
                    : self.args.per_device_eval_batch_size, :
                ]
                pad_token_id = self.pad_token_id # Using the property which accesses self.tokenizer.pad_token_id
                preds_emb = self.call_embedding_model(
                    input_ids=preds_sample_retokenized,
                    attention_mask=(preds_sample_retokenized != pad_token_id).to(
                        self.args.device
                    ),
                )
                preds_sample_labels_retokenized = self.embedder_tokenizer(
                    decoded_labels, padding=True, truncation=False, return_tensors="pt"
                )["input_ids"].to(preds_sample.device)
                preds_sample_labels_retokenized = preds_sample_labels_retokenized[
                    : self.args.per_device_eval_batch_size, :
                ]
                labels_emb = self.call_embedding_model(
                    input_ids=preds_sample_labels_retokenized,
                    attention_mask=(preds_sample_labels_retokenized != pad_token_id).to(
                        self.args.device
                    ),
                )
                emb_cos_sims = torch.nn.CosineSimilarity(dim=1)(preds_emb, labels_emb)
                emb_topk_equal = (
                    (preds_emb[:, :32000].argmax(1) == labels_emb[:, :32000].argmax(1))
                    .float()
                    .cpu()
                )
                sim_result = {
                    "emb_cos_sim": emb_cos_sims.mean().item(),
                    "emb_cos_sim_sem": sem(emb_cos_sims.cpu().numpy()),
                    "emb_top1_equal": emb_topk_equal.mean().item(),
                    "emb_top1_equal_sem": sem(emb_topk_equal),
                }

        except (TypeError, RuntimeError) as e:
            logger.error(f"Error computing embedding similarity: {e}")
            sim_result = {"emb_cos_sim": 0, "emb_cos_sim_sem": 0, "emb_top1_equal": 0, "emb_top1_equal_sem": 0}


        # Store stuff for access later.
        # self.preds_emb = preds_emb.cpu()
        # self.labels_emb = labels_emb.cpu()
        # Storing cleaned lists might be more appropriate if they are used later
        self.preds_sample_list = cleaned_preds_sample_list 
        self.preds_sample_labels_list = self._clean_token_ids(preds_sample_labels_list) if not isinstance(preds_sample_labels_list[0], tuple) else [
            (self._clean_token_ids([item[0]])[0], self._clean_token_ids([item[1]])[0]) for item in preds_sample_labels_list
        ] # Clean based on structure

        metrics = {**num_tokens_metrics, **bleu_result, **sim_result}
        return metrics
