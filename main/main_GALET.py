import os

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import time
import pickle
import scipy
import numpy as np
import torch
import torch.nn as nn
import argparse
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
import gc


############################################### Preliminaries ###############################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offset', type=int, default=1, help='Offset to distinguish a saved file')
    args = parser.parse_args()
    return args


args = parse_args()

# configuration
save_file = 1

seeds = [1]
num_run = len(seeds)
num_step = 2000

# learning parameters
batch_size = 32
alpha = 0.00001
beta = 0.00001
N = 5
T = 5
rho = 0.1

# x_net
num_T = 2

print("\n" + "=" * 80)
print(f"STARTING NEW EXPERIMENT SET FOR GALET")
print("=" * 80 + "\n")

base_file_name = "GALET"
print(base_file_name)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_type = torch.bfloat16
print(device)
print()

config_lora = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules='all-linear'
)


def build_x(dim: int):
    layers = []
    layers.append(nn.Linear(1, dim, bias=False))
    layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)


def get_outputs(y_opt, x_opt, y_model, x_model, inputs_y_ids, input_y_masks, one_tensor, mask_tensor):
    if y_opt: y_opt.zero_grad()
    if x_opt: x_opt.zero_grad()
    outputs_y = y_model(input_ids=inputs_y_ids.to(y_model.device), attention_mask=input_y_masks.to(y_model.device))
    outputs_x = x_model(one_tensor, mask_tensor)
    return outputs_y, outputs_x


def get_output(y_opt, x_opt, y_model, inputs_y_ids, input_y_masks):
    if y_opt: y_opt.zero_grad()
    if x_opt: x_opt.zero_grad()
    outputs_y = y_model(input_ids=inputs_y_ids.to(y_model.device), attention_mask=input_y_masks.to(y_model.device))
    return outputs_y


class x_model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.x_net = build_x(dim)

    def forward(self, one, idx):
        weights = self.x_net(one)
        return torch.gather(weights, 1, idx)


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_idx):
        super().__init__()
        self.entropy_net = torch.nn.CrossEntropyLoss(ignore_index=ignore_idx)

    def forward(self, inputs, targets, weights):
        c_entropy = self.entropy_net(inputs, targets)
        return torch.mean(weights * c_entropy)


ignore_idx = -100


################################################## Dataset ##################################################

def sort_data(dataset, max_size):
    idx = []
    prompt = []
    response = []
    for i, a in enumerate(dataset):
        if len(a['prompt']) + len(a['response']) < max_size:
            idx.append(i)
            prompt.append(a['prompt'])
            response.append(a['response'])
    return idx, prompt, response


def get_scores(dataset):
    data = []
    for a in dataset.features:
        data.append(dataset[a])
    return np.array(data[2:])


LLM_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(LLM_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

ds_tra = load_dataset("nvidia/HelpSteer")['train']
ds_val = load_dataset("nvidia/HelpSteer")['validation']

idx_tra, prompt_tra, response_tra = sort_data(ds_tra, 1000)
idx_val, prompt_val, response_val = sort_data(ds_val, 1000)

scores_tra = get_scores(ds_tra)[:, np.array(idx_tra)]
scores_val = get_scores(ds_val)[:, np.array(idx_val)]

# Train
scores_tra_high_idx = np.where(np.average(scores_tra, axis=0) > 2.5)[0]
scores_tra_low_idx = np.where(np.average(scores_tra, axis=0) <= 2)[0]

templates = []
for idx_list in [scores_tra_high_idx, scores_tra_low_idx]:
    templates_temp = []
    for idx in idx_list:
        chat_template = []
        chat_template.append({"role": "system", "content": "You are a chatbot who answers a given question."})
        chat_template.append({"role": "user", "content": prompt_tra[idx]})
        chat_template.append({"role": "assistant", "content": response_tra[idx]})
        templates_temp.append(chat_template)
    templates.append(templates_temp)
template_tra = templates[0] + templates[1]
train_num = np.ones([len(templates[0]) + len(templates[1]), 1])
train_num[:len(template_tra[0])] = 0

train_input = []
train_label = []
input_maxlen = 0
for a in range(len(template_tra)):
    prompt = tokenizer.apply_chat_template(template_tra[a][:-1], tokenize=False, add_generation_prompt=True).replace(
        tokenizer.bos_token, "")
    output_p = tokenizer(prompt, padding=False, max_length=64)
    prompt_id = output_p.input_ids

    response = tokenizer.apply_chat_template(template_tra[a], tokenize=False, add_generation_prompt=False).replace(
        tokenizer.bos_token, "")
    output_r = tokenizer(response[len(prompt):], padding=False, max_length=64)
    response_id = output_r.input_ids[1:]

    input_id = prompt_id + response_id
    label_id = [-100] * len(prompt_id) + response_id

    if len(input_id) > input_maxlen:
        input_maxlen = len(input_id)

    train_input.append(input_id)
    train_label.append(label_id)

maxlen = input_maxlen + 5
train_mask = []
for a in range(len(template_tra)):
    input_id = train_input[a]
    label_id = train_label[a]

    input_id = input_id + [tokenizer.pad_token_id] * (maxlen - len(input_id))
    label_id = label_id + [-100] * (maxlen - len(label_id))
    attention_mask = [1 if t != tokenizer.pad_token_id else 0 for t in input_id]

    train_input[a] = input_id
    train_label[a] = label_id
    train_mask.append(attention_mask)

train_input = torch.as_tensor(np.array(train_input))
train_label = torch.as_tensor(np.array(train_label))
train_mask = torch.as_tensor(np.array(train_mask))

# Test
template_val = []
for a in range(len(prompt_val)):
    chat_template = []
    chat_template.append({"role": "system", "content": "You are a chatbot who answers a given question."})
    chat_template.append({"role": "user", "content": prompt_val[a]})
    chat_template.append({"role": "assistant", "content": response_val[a]})
    template_val.append(chat_template)

test_input = []
test_label = []
test_mask = []
for a in range(len(template_val)):
    prompt = tokenizer.apply_chat_template(template_val[a][:-1], tokenize=False, add_generation_prompt=True).replace(
        tokenizer.bos_token, "")
    output_p = tokenizer(prompt, padding=False, max_length=64)
    prompt_id = output_p.input_ids

    response = tokenizer.apply_chat_template(template_val[a], tokenize=False, add_generation_prompt=False).replace(
        tokenizer.bos_token, "")
    output_r = tokenizer(response[len(prompt):], padding=False, max_length=64)
    response_id = output_r.input_ids[1:]

    input_id = prompt_id + response_id
    label_id = [-100] * len(prompt_id) + response_id

    input_id = input_id + [tokenizer.pad_token_id] * (maxlen - len(input_id))
    label_id = label_id + [-100] * (maxlen - len(label_id))
    attention_mask = [1 if t != tokenizer.pad_token_id else 0 for t in input_id]

    test_input.append(input_id)
    test_label.append(label_id)
    test_mask.append(attention_mask)

test_input = torch.as_tensor(np.array(test_input))
test_label = torch.as_tensor(np.array(test_label))
test_mask = torch.as_tensor(np.array(test_mask))

# tensor
train_num = torch.as_tensor(train_num).long()
one_tensor = torch.as_tensor(np.ones([batch_size, 1])).float()

del ds_tra, ds_val
print()

train_loss_chunk = []
test_loss_chunk = []
soft_chunk = []
times_chunk = []
ll_grad_norm_chunk = []

################################################## Running ##################################################

for run in range(num_run):

    seed = seeds[run]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(seed)

    print(f"Run {run + 1} / {num_run}")
    main_model = AutoModelForCausalLM.from_pretrained(LLM_name, device_map="auto", torch_dtype=data_type)
    main_model.config.pad_token_id = tokenizer.bos_token_id
    main_model = get_peft_model(main_model, config_lora)
    main_model.config.use_cache = False
    main_model.gradient_checkpointing_enable()
    main_model.print_trainable_parameters()
    lora_params = [p for p in main_model.parameters() if p.requires_grad]

    weighting_model = x_model(num_T).to(device)
    weighting_model.state_dict()['x_net.0.weight'].fill_(0)

    f_loss_MSE = torch.nn.CrossEntropyLoss(ignore_index=ignore_idx)
    g_loss_WMSE = WeightedCrossEntropyLoss(ignore_idx=ignore_idx)

    train_loss_t = []
    test_loss_t = []
    soft_t = []
    times_t = []
    ll_grad_norm_t = []

    for param in weighting_model.parameters():
        x_param = param.tolist()
        soft_t.append(np.squeeze(scipy.special.softmax(x_param)))

    for t in range(num_step):
        batch_idx_train = np.random.choice(len(train_input), batch_size, replace=False)
        batch_train_input = train_input[batch_idx_train, :]
        batch_train_label = train_label[batch_idx_train, :]
        batch_train_mask = train_mask[batch_idx_train, :]
        train_mask_tensor = train_num[batch_idx_train]

        batch_idx_test = np.random.choice(len(test_input), batch_size, replace=False)
        batch_test_input = test_input[batch_idx_test, :]
        batch_test_label = test_label[batch_idx_test, :]
        batch_test_mask = test_mask[batch_idx_test, :]

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            start_time = time.time()

            for n in range(N):
                outputs_y, outputs_x = get_outputs(None, None, main_model, weighting_model,
                                                   batch_train_input, batch_train_mask, one_tensor.to(device),
                                                   train_mask_tensor.to(device))
                logits_flat = outputs_y.logits.view(-1, outputs_y.logits.size(-1)).to(device)
                labels_flat = batch_train_label.view(-1).long().to(device)
                g_loss = g_loss_WMSE(logits_flat, labels_flat, outputs_x)

                g_y = torch.autograd.grad(g_loss, lora_params)

                with torch.no_grad():
                    for p, gy in zip(lora_params, g_y):
                        p.sub_(beta * gy)

            outputs_y, outputs_x = get_outputs(None, None, main_model, weighting_model,
                                               batch_train_input, batch_train_mask, one_tensor.to(device),
                                               train_mask_tensor.to(device))
            logits_flat = outputs_y.logits.view(-1, outputs_y.logits.size(-1)).to(device)
            labels_flat = batch_train_label.view(-1).long().to(device)
            g_loss = g_loss_WMSE(logits_flat, labels_flat, outputs_x)

            g_y = torch.autograd.grad(g_loss, lora_params, create_graph=True)

            outputs_y_val = get_output(None, None, main_model, batch_test_input, batch_test_mask)
            logits_val_flat = outputs_y_val.logits.view(-1, outputs_y_val.logits.size(-1)).to(device)
            labels_val_flat = batch_test_label.view(-1).long().to(device)
            f_loss = f_loss_MSE(logits_val_flat, labels_val_flat)

            f_y = torch.autograd.grad(f_loss, lora_params)

            def HVP(v_tuple):
                v_detached = [v_i.detach() for v_i in v_tuple]
                dot = sum(torch.sum(gy * v_i).to(device) for gy, v_i in zip(g_y, v_detached))
                return torch.autograd.grad(dot, lora_params, retain_graph=True)

            w = [torch.zeros_like(p) for p in lora_params]

            for t_step in range(T):
                H_yy_w = HVP(w)
                v = [fy_i + hyy_wi for fy_i, hyy_wi in zip(f_y, H_yy_w)]
                d_w = HVP(v)
                w = [w_i - rho * dw_i for w_i, dw_i in zip(w, d_w)]

            g_y_w_dot = sum(torch.sum(gy * w_i.detach()).to(device) for gy, w_i in zip(g_y, w))
            x_params = list(weighting_model.parameters())
            d_x = torch.autograd.grad(g_y_w_dot, x_params)

            with torch.no_grad():
                for param, dx in zip(x_params, d_x):
                    param.sub_(alpha * dx)

            end_time = time.time()
            elapsed_time = end_time - start_time
            times_t.append(elapsed_time)

        with torch.no_grad():
            train_loss_t.append(g_loss.item())
            test_loss_t.append(f_loss.item())

            current_ll_grad_norm = np.sqrt(sum(torch.sum(gy ** 2).to(device).item() for gy in g_y))
            ll_grad_norm_t.append(current_ll_grad_norm)

            for param in weighting_model.parameters():
                x_param = param.tolist()
                soft_t.append(np.squeeze(scipy.special.softmax(x_param)))

            if (t + 1) % 1 == 0:
                print(
                    f"Finished {t + 1} of {num_step} steps | Train Loss: {g_loss.item():.4f} | Val Loss: {f_loss.item():.4f} | LL Grad Norm: {current_ll_grad_norm:.4f}")

        del outputs_x, outputs_y, outputs_y_val, g_loss, f_loss
        del f_y, g_y, w, v, H_yy_w, d_w, d_x, g_y_w_dot
        current_ll_grad_norm = None

    train_loss_chunk.append(train_loss_t)
    test_loss_chunk.append(test_loss_t)
    soft_chunk.append(soft_t)
    times_chunk.append(times_t)
    ll_grad_norm_chunk.append(ll_grad_norm_t)
    print(f"(Elapsed time: {sum(times_t):.3f} seconds)")

    if (run + 1) % 1 == 0 and save_file:
        current_seed = seeds[run]
        chunk_file_name = f"{base_file_name}_seed{current_seed}"
        save_path = f'{chunk_file_name}.pkl'

        print(f"--- Run {run + 1}: Saving results for seed {current_seed}... ---")
        items_to_save = [train_loss_chunk, test_loss_chunk, soft_chunk, times_chunk, ll_grad_norm_chunk]
        with open(save_path, 'wb') as file:
            pickle.dump(items_to_save, file)

        print(f"--- Success! Saved to: {save_path} ---")

        train_loss_chunk = []
        test_loss_chunk = []
        soft_chunk = []
        times_chunk = []
        ll_grad_norm_chunk = []

    try:
        del logits_flat, labels_flat
        del logits_val_flat, labels_val_flat
    except NameError:
        pass

    del main_model
    del weighting_model
    del lora_params

    del f_loss_MSE
    del g_loss_WMSE

    try:
        del batch_train_input, batch_train_label, batch_train_mask, train_mask_tensor
        del batch_test_input, batch_test_label, batch_test_mask
    except NameError:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print("--- GPU memory cleared. Proceeding to next run. ---")
    print("\n" + "=" * 50 + "\n")
