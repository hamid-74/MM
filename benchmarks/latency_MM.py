"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys
sys.path.append("/src/")

from tqdm import tqdm

from mm import MM



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=0,
        choices=[0, 1],
        help="0: exeute at GPU (baseline), 1: offload to CPU.",
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
        default=40,
        help="Number of tokens to generate.",
    )

    args = parser.parse_args()

    path_json = "ShareGPT_V3_unfiltered_cleaned_split.json"
    with open(path_json, "r") as f:
        data = json.load(f)

    texts = []
    for d in data:
        if len(d["conversations"]) == 0:
            continue
        # the input of the first round
        texts.append(" ".join(d["conversations"][0]["value"].split()))

    random.seed(0)
    random.shuffle(texts)
    model = MM(args)
    n_sample = 10


    input_tokens = [16, 32, 64, 128]
    output_tokens = [16, 32, 64, 128, 256, 512]
    throughputs = dict()


    for input_token in tqdm(input_tokens, desc="Input Tokens"):
        for output_token in tqdm(output_tokens, desc="Output Tokens", leave=False):
            idx_text = 0
            prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
            for _ in tqdm(range(n_sample)):
                while True:
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        # enough input length
                        break
                prefill_time, decode_time, hit_rate = model.generate(
                    text, output_token=output_token, input_token=input_token
                )
                # print(f"prefill time: {prefill_time}, decode_time:{decode_time}")
                prefill_time_sum += prefill_time
                decode_time_sum += decode_time
                hit_rate_sum += hit_rate
            
            avg_prefill_time = prefill_time_sum / n_sample
            avg_decode_time = decode_time_sum / n_sample
            print(
                f"input_token: {input_token}, output_token: {output_token}, "
                f"prefill_time: {prefill_time_sum / n_sample}, "
                f"decode_time: {decode_time_sum / n_sample}, "
                f"hit_rate: {hit_rate_sum / n_sample},"
                f"{output_token / (avg_prefill_time + avg_decode_time):.2f}token/s"
            )
            key = (input_token, output_token)
            key_str = '_'.join(map(str, key))
            throughputs[key_str] = output_token / (avg_prefill_time + avg_decode_time)
            file_path = "../results/mm_throughput.json"

            # Dump the dictionary to the file
            with open(file_path, "w") as json_file:
                json.dump(throughputs, json_file)
