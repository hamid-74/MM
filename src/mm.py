import copy
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from transformers import BitsAndBytesConfig
from transformers.modeling_attn_mask_utils import *
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)

class MM:
    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        self.model = transformers.MixtralForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            # device_map='cpu',
            use_cache=True,

        )
        self.model_4 = transformers.MixtralForCausalLM.from_pretrained(
            args.model,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,

        )

        print(f"Total memory allocated on GPU: {torch.cuda.memory_allocated()} bytes")
        print(f"free mem after loading 4bit model: {torch.cuda.get_device_properties(self.dev).total_memory - torch.cuda.memory_allocated()} bytes")
        self.lm_head = self.model.lm_head
        self.model = self.model.model
        self.model_4 = self.model_4.model
        
        self.expert_placeholder = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        self.cpu_offload = args.cpu_offload

        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)
        all_expert_positions = [(i, j) for i in range(self.n_layer) for j in range(self.n_expert)]

        self.popular_experts = random.sample(all_expert_positions, 256)


        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        self.bring_non_expert_to_gpu()

        # 0: CPU, 1: GPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)

        # 0:b16flaot, 1: 4bitfloat
        self.expert_precision = np.zeros((self.n_layer, self.n_expert), dtype=int)

        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        if self.cpu_offload == 0:
            no_expert_16, no_expert_4 = self.calc_prec_dist_on_gpu()
        elif self.cpu_offload == 2:
            self.no_expert_4 = args.no_expert_4
            no_expert_16 = 256 - self.no_expert_4
            no_expert_16_on_gpu, _ = self.calc_prec_dist_on_gpu()
        


        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}"
        )
        print(f"no_expert_16:{no_expert_16}, no_4:{self.no_expert_4}, no_expert_16_on_gpu:{no_expert_16_on_gpu}")

        if self.cpu_offload == 0:
            self.set_expert_loc(n_expert_on_gpu)
            self.set_expert_precision(no_expert_16)
        elif self.cpu_offload == 2:
            self.set_expert_precision(no_expert_16)
            self.set_expert_loc(no_expert_16_on_gpu)

        if self.cpu_offload == 0:
            self.mm_bring_expert_to_gpu()
        elif self.cpu_offload == 2:
            self.bring_expert_to_gpu()

        self.generated_tokens = []
        self.generated_logits = []

        print(f"Total memory allocated on GPU: {torch.cuda.memory_allocated()} bytes")

        print("Model is ready.")

    def bring_non_expert_to_gpu(self):
        """Bring non-expert layers to GPU"""
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        for i in range(len(self.model.layers)):
            self.model.layers[i].self_attn.to(self.dev)
            self.model.layers[i].input_layernorm.to(self.dev)
            self.model.layers[i].block_sparse_moe.gate.to(self.dev)
            self.model.layers[i].post_attention_layernorm.to(self.dev)
            # only model.layers[i].block_sparse_moe.experts is on CPU

    def set_expert_loc(self, n_expert_on_gpu):

        for i in range(n_expert_on_gpu):
            i_layer, i_expert = self.popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1


    def set_expert_precision(self, no_expert_16):
        """Set the precision of experts"""

        for i in range(self.n_layer * self.n_expert):
            i_layer, i_expert = self.popular_experts[i]
            if(i<no_expert_16):
                self.expert_precision[i_layer, i_expert] = 1
            else:
                self.expert_precision[i_layer, i_expert] = 0

    def bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU"""
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.model.layers[i].block_sparse_moe.experts[j].to(self.dev)

    def mm_bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU"""
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_16(i, j):
                    self.model.layers[i].block_sparse_moe.experts[j].to(self.dev)
                else:
                    # print("i:{}, j:{}".format(i, j))
                    self.model_4.layers[i].block_sparse_moe.experts[j].to(self.dev)


    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU"""
        return self.expert_loc[i_layer, i_expert] == 1
    
    def is_expert_16(self, i_layer, i_expert):
        """Determine if the expert is 16 bit or 4 bit"""
        return self.expert_precision[i_layer, i_expert] == 1

    def calc_n_expert_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        # get the number of parameters of one expert
        n_param = sum(
            p.numel()
            for p in self.model.layers[0].block_sparse_moe.experts[0].parameters()
        )
        # get the amount of free memory on GPU
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        free_mem = total_mem * 0.60 - torch.cuda.memory_allocated(self.dev)
        return int((free_mem) // (n_param * 2))


    def calc_prec_dist_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        # get the number of parameters of one expert
        n_param = sum(
            p.numel()
            for p in self.model.layers[0].block_sparse_moe.experts[0].parameters()
        )

        print(f"number of parameters in an expert: {n_param}")
        # get the amount of free memory on GPU
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        free_mem = total_mem * 0.60 - torch.cuda.memory_allocated(self.dev)

        print(f"free mem: {free_mem}")

        
        # free_mem = total_mem * 0.5 - torch.cuda.memory_allocated(self.dev)


        expert_mem_16 = n_param * 2
        expert_mem_4 = int(expert_mem_16 / 4)

        # 2 meaning 2 bytes for each b16float number
        # no_full = int((free_mem - 256*expert_mem_4)/(expert_mem_16-expert_mem_4))
        # no_4 = self.n_layer * self.n_expert - no_full

        # no_full = int((free_mem)/(expert_mem_16)) - int((self.no_expert_4)/4) - 44    $$$$$ this line is for benchmarking throughput sweep
        no_full = int((free_mem)/(expert_mem_16))
        no_4 = 0

        # no_full = int((free_mem)/(expert_mem_16))
        # no_4 = 0
        
        
        return no_full, no_4

    def generate(self, text, output_token=40, input_token=None):
        self.generated_logits.clear()
        self.generated_tokens.clear()
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        input_ids, position_ids = self.tokenize(text)

        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]

        tick = time.time()
        is_decode = False
        prefill_time, decode_time = 0, 0
        for i_token in range(output_token):


            # tick = time.time()
            
            # print(self.tokenizer.decode(input_ids[0, :]))
            logits = self.mixtral_forward(
                input_ids,
                position_ids,
                is_decode,
            )
            # print('Time:', time.time() - tick)

            logits = logits.to("cpu")

            output = torch.argmax(logits, dim=-1)
            self.past_key_values_length += output.shape[-1]
            input_ids = output[:, -1].unsqueeze(0).to(self.dev)
            position_ids = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + 1,
                dtype=torch.long,
                device=self.dev,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, 1)
            if not is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True
        decode_time = time.time() - tick

        return prefill_time, decode_time, self.cnt_expert_hit

    def generate_from_tokens(self, input_ids, output_token=20, input_token=None):
        self.generated_logits.clear()
        self.generated_tokens.clear()

        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        input_ids, position_ids = self.tokenize_input_ids(input_ids)

        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]

        tick = time.time()
        is_decode = False
        prefill_time, decode_time = 0, 0
        for i_token in range(output_token):
            # tick = time.time()
            # print(self.tokenizer.decode(input_ids[0, :]))
            
            self.generated_tokens.append(input_ids[0, :])
            
            
            logits = self.mixtral_forward(
                input_ids,
                position_ids,
                is_decode,
            )

            self.generated_logits.append(logits)
            # print('Time:', time.time() - tick)

            logits = logits.to("cpu")

            output = torch.argmax(logits, dim=-1)
            self.past_key_values_length += output.shape[-1]
            input_ids = output[:, -1].unsqueeze(0).to(self.dev)
            position_ids = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + 1,
                dtype=torch.long,
                device=self.dev,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, 1)
            if not is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True
        decode_time = time.time() - tick
        
        # return prefill_time, decode_time, self.cnt_expert_hit / self.cnt_expert_all, self.generated_tokens
        return prefill_time, decode_time, self.cnt_expert_hit, self.generated_tokens, self.generated_logits


    def tokenize(self, text):
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.dev)
        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.dev
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])
        return input_ids, position_ids

    def tokenize_input_ids(self, input_ids):

        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.dev
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])
        return input_ids, position_ids

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, is_decode):
        hidden_dim = self.model.config.hidden_size
        inps = input_ids.to(self.dev)
        inps = self.model.embed_tokens(inps)

        for i_layer, layer in enumerate(self.model.layers):
            original_inps_shape = inps.shape

            inps_residual = inps
            inps = layer.input_layernorm(inps)



            attention_mask = _prepare_4d_causal_attention_mask(
                None,
                (1, inps.shape[1]),
                inps,
                0,
                sliding_window=self.model.config.sliding_window,
            )
            inps, self_attn_weights, present_key_value = layer.self_attn(
                inps,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
                attention_mask=attention_mask,
            ) 

            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)

            inps = inps.view(-1, hidden_dim)
            router_logits = layer.block_sparse_moe.gate(inps)
            routing_weights = F.softmax(router_logits, dim=1)
            routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # intermediate variable to store the output of experts
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.block_sparse_moe.experts

            ######
            experts_4 = self.model_4.layers[i_layer].block_sparse_moe.experts


            if self.cpu_offload == 0:

                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

                for i_expert in range(len(experts)):
                    # is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                    is_16 = self.is_expert_16(i_layer, i_expert)
                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:
                        # print(f"Expert {i_expert}: has no tokens")
                        continue

                    torch.cuda.synchronize()
                    top_2_list = top_2.tolist()
                    idx_list = idx.tolist()

                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)

                    if not is_16:

                        # print("i_layer:{}, i_expert:{}".format(i_layer, i_expert))

                        self.cnt_expert_all = self.cnt_expert_all + 1
                        
                        current_state = experts_4[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )

                    else:
                        # print("i_layer:{}, i_expert:{}".format(i_layer, i_expert))

                        self.cnt_expert_all = self.cnt_expert_all + 1
                        self.cnt_expert_hit = self.cnt_expert_hit + 1
                        
                        current_state = experts[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )

                    inps_after_experts.index_add_(
                        0, top_2, current_state.to(inps.dtype)
                    )


                    # end of one expert

            elif self.cpu_offload == 2:
                # sweep mem vs ppl
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

                for i_expert in range(len(experts)):
                    is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                    is_16 = self.is_expert_16(i_layer, i_expert)

                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:
                        # print(f"Expert {i_expert}: has no tokens")
                        continue

                    # torch.cuda.synchronize()
                    top_2_list = top_2.tolist()
                    idx_list = idx.tolist()

                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)

                    if not is_16:

                        # print("i_layer:{}, i_expert:{}".format(i_layer, i_expert))

                        self.cnt_expert_all = self.cnt_expert_all + 1
                        
                        current_state = experts_4[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    elif is_16 and (not is_cuda):
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder(
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    elif is_16 and is_cuda:
                        current_state = experts[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    else:
                        print("error!!!!!")

                    inps_after_experts.index_add_(
                        0, top_2, current_state.to(inps.dtype)
                    )

                    if not is_cuda:
                        experts[i_expert] = experts[i_expert].to(
                            "cpu", non_blocking=True
                        )

                    # end of one expert

            # addition because there's residual connection over moe layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)

            # end of one layer

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logis

