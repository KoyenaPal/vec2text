import os
os.environ["DISABLE_FLASH_ATTN"] = "1"
import argparse
import transformers
from vec2text.experiments import experiment_from_args
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="t5-base")
    parser.add_argument("--embedder", type=str, default="EleutherAI/pythia-14m")
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--from_gradients", action="store_true")
    parser.add_argument("--tinydata", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true", default=False)
    # REMEMBER TO CHANGE THIS
    parser.add_argument("--output_dir", type=str, default="person_finder_inverter_models")
    parser.add_argument("--wandb_exp_name", type=str, default="initial_run_all_grads")
    parser.add_argument("--all_grads", action="store_true", default=False)
    parser.add_argument("--embed_in_grads", action="store_true", default=False)
    parser.add_argument("--embed_out_grads", action="store_true", default=False)
    parser.add_argument("--layer_0_grads", action="store_true", default=False)
    parser.add_argument("--layer_1_grads", action="store_true", default=False)
    parser.add_argument("--layer_2_grads", action="store_true", default=False)
    parser.add_argument("--layer_3_grads", action="store_true", default=False)
    parser.add_argument("--layer_4_grads", action="store_true", default=False)
    parser.add_argument("--layer_5_grads", action="store_true", default=False)
    parser.add_argument("--reduction_version_SVD", action="store_true", default=True)
    parser.add_argument("--reduction_version_JL", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--eval_model", action="store_true", default=False)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    # look at eval_steps
    hidden_data_suffix = "__hidden" if args.hidden else ""
    tiny_data_suffix = "__tinydata" if args.tinydata else ""
    from_gradients_suffix = "__from_gradients" if args.from_gradients else ""
    all_grads_suffix = "__all_grads" if args.all_grads else ""
    embed_in_grads_suffix = "__embed_in" if args.embed_in_grads else ""
    embed_out_grads_suffix = "__embed_out" if args.embed_out_grads else ""
    layer_0_grads_suffix = "__layer_0" if args.layer_0_grads else ""
    layer_1_grads_suffix = "__layer_1" if args.layer_1_grads else ""
    layer_2_grads_suffix = "__layer_2" if args.layer_2_grads else ""
    layer_3_grads_suffix = "__layer_3" if args.layer_3_grads else ""
    layer_4_grads_suffix = "__layer_4" if args.layer_4_grads else ""
    layer_5_grads_suffix = "__layer_5" if args.layer_5_grads else ""
    reduction_version_SVD_suffix = "__reduction_version_SVD" if args.reduction_version_SVD else ""
    reduction_version_JL_suffix = "__reduction_version_JL" if args.reduction_version_JL else ""
    
    use_less_data = 1000 if args.tinydata else -1
    wandb_exp_name = args.wandb_exp_name
    if not args.all_grads:
        wandb_exp_name = args.wandb_exp_name + hidden_data_suffix + tiny_data_suffix + from_gradients_suffix + all_grads_suffix + embed_in_grads_suffix + embed_out_grads_suffix + layer_0_grads_suffix + layer_1_grads_suffix + layer_2_grads_suffix + layer_3_grads_suffix + layer_4_grads_suffix + layer_5_grads_suffix + reduction_version_SVD_suffix + reduction_version_JL_suffix
    if args.all_grads:
        args.embed_in_grads = True
        args.embed_out_grads = True
        args.layer_0_grads = True
        args.layer_1_grads = True
        args.layer_2_grads = True
        args.layer_3_grads = True
        args.layer_4_grads = True
        args.layer_5_grads = True
        args.reduction_version_SVD = True
    model_args = ModelArguments(
        model_name_or_path=args.model,
        embedder_model_name=args.embedder,
        use_frozen_embeddings_as_input=False,
        # embeddings_from_layer_n=embeddings_from_layer_n,
        embedder_no_grad=(not args.from_gradients),
        reduction_version_SVD=args.reduction_version_SVD,
        reduction_version_JL=args.reduction_version_JL,
        embed_in_gradient=args.embed_in_grads,
        embed_out_gradient=args.embed_out_grads,
        layer_0_gradient=args.layer_0_grads,
        layer_1_gradient=args.layer_1_grads,
        layer_2_gradient=args.layer_2_grads,
        layer_3_gradient=args.layer_3_grads,
        layer_4_gradient=args.layer_4_grads,
        layer_5_gradient=args.layer_5_grads,
    )
    data_args = DataArguments(
        dataset_name="person_finder",
        use_less_data=use_less_data,
        )
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        experiment=("inversion_from_gradients" if args.from_gradients else "inversion_from_logits"),
        use_wandb=args.wandb,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
        run_name=wandb_exp_name
    )
    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.run()


if __name__ == "__main__":
    main()
