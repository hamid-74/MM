import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm

from datasets import load_dataset
import argparse

from mm import MM

import os

import json


dataset_samples = 128

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_gpu_exec_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=2,
        choices=[0, 2],
        help="0: MM, 2: mem vs ppl sweep",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="University of Washington is",
        help="Input text to generate.",
    )
    parser.add_argument(
        "--n-token",
        type=int,
        default=30,
        help="Number of tokens to generate.",
    )
    parser.add_argument(
        "--no-expert-4",
        type=int,
        default=0,
        help="Number of experts in 4 precision",
    )


    return parser


class Custom_Evaluator:
    def __init__(self, dataset, column, tokenizer, device, n_samples=dataset_samples):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device



        self.dataset = tokenizer(
            "\n\n".join(dataset[column]), return_tensors="pt"
        ).input_ids.to(device)


        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):

        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            batch_input_ids = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.dev)

            with torch.no_grad():
                # lm_logits = model(batch).logits
                prefill_time, decode_time, hit_rate, generated_tokens, lm_logits = model.generate_from_tokens(batch_input_ids, output_token=1)

            lm_logits = lm_logits[0]

            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))

file_path = "../results/mem_vs_ppl.json"
if os.path.exists(file_path):
    with open(file_path, 'r') as json_file:
        sweep_ppl = json.load(json_file)
else:
    sweep_ppl = dict()
    
model_id = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)



for no_expert_4 in tqdm.tqdm(range(0,257), desc="Sweep 4 bit Experts:"): 


    
    dataset_labels = ['wikitext', 'ptb_text_only', 'c4']
    # dataset_labels = [ 'ptb_text_only', 'c4']
    # dataset_labels = ['c4']

        
    parser = generate_gpu_exec_parser()
    args = parser.parse_args()
    args.no_expert_4 = no_expert_4

    model = MM(args)

    for label in dataset_labels: 
        if label == 'wikitext':
            dataset = load_dataset('wikitext', split='test')

        elif label == 'ptb_text_only':
            dataset = load_dataset('ptb_text_only', 'default', split='train')

        elif label == 'c4':
            dataset = load_dataset('allenai_c4', split='validation')
            dataset = dataset.select(range(600))


        if label == 'wikitext' or label == 'c4':
            evaluator = Custom_Evaluator(dataset, "text", tokenizer, "cuda")
        elif label == 'ptb_text_only':
            evaluator = Custom_Evaluator(dataset, "sentence", tokenizer, "cuda")


        ppl = evaluator.evaluate(model)
        key_str = f"{label}_{no_expert_4}"
        
        sweep_ppl[key_str] = float(ppl)
        print(f"dataset: {label}, no_expert_4: {no_expert_4},  ppl: {ppl}")


        

        # Dump the dictionary to the file
        with open(file_path, "w") as json_file:
            json.dump(sweep_ppl, json_file)


    del model
    del evaluator
    torch.cuda.empty_cache()


