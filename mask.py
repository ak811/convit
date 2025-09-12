import torch
import einops
import numpy as np
import random 
import math


def get_mask_attn(mask_ratio,batch_size,heads,num_patches):
    n_dim = batch_size*heads
    di = num_patches*num_patches
    k = int(mask_ratio * di)
    sample =  torch.rand(n_dim, di,device="cuda").topk(k, dim=1).indices
    mask = torch.ones(n_dim, di, dtype=torch.bool,device="cuda")
    mask.scatter_(dim=1, index=sample, value=False)
    masked_tensor = einops.rearrange(mask, '(b h) (n d) -> b h n d', b=batch_size, n=num_patches, h=heads , d= num_patches)
    return masked_tensor 

def token_pruner(x, keep_ratio = 0.95):
  x_ = x[:, 1:]
  bsz = x_.shape[0]
  num_tokens = x_.shape[1]
  x_ = x_.reshape(bsz*num_tokens, -1)
  keep_num_tokens_per_sample = int(num_tokens * keep_ratio)
  keep_num_tokens = bsz * keep_num_tokens_per_sample
  keep_token_ind = torch.rand((bsz * num_tokens),device='cuda').topk(keep_num_tokens).indices
  new_x_ = torch.index_select(x_, dim=0, index=keep_token_ind).reshape(bsz, keep_num_tokens_per_sample, -1)
  new_x = torch.cat((x[:,0].unsqueeze(1), new_x_), dim=1)

  return new_x   

def get_mask_attn_sampling(q, k):
  r = 4
  q_adjusted = q[:, :, 1:, :]
  k_adjusted = k[:, :, 1:, :]

  B, H, N, D = q_adjusted.size()

  #sample =  torch.rand(B*H, N*N,device=q.device).topk(N, dim=-1).indices
  mask = torch.zeros(B*H, N*N, device=q.device)
  #mask.scatter_(dim=-1, index=sample, value=1)
  #mask = einops.rearrange(mask, '(b h) (n d) -> b h n d', b=B, n=N, h=H , d=N)
  mask = mask.view(B,H,N,N)
 
  for i in range(-r, r + 1):
    if i != 0 :
      indices = torch.arange(max(-i, 0), min(N, N - i))
      mask[:, :, indices, indices + i] = 1
  mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device)
  mask_extended[:, :, 1:, 1:] = mask
  return mask_extended     

def get_mask_attn_sampling_bigbird(q, k):
  r = 4
  q_adjusted = q[:, :, 1:, :]
  k_adjusted = k[:, :, 1:, :]

  B, H, N, D = q_adjusted.size()

  sample =  torch.rand(B*H, N*N,device=q.device).topk(N, dim=-1).indices
  mask = torch.zeros(B*H, N*N, device=q.device)
  mask.scatter_(dim=-1, index=sample, value=1)
  #mask = einops.rearrange(mask, '(b h) (n d) -> b h n d', b=B, n=N, h=H , d=N)
  mask = mask.view(B,H,N,N)
 
  for i in range(-r, r + 1):
    indices = torch.arange(max(-i, 0), min(N, N - i))
    mask[:, :, indices, indices + i] = 1

  mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device)
  mask_extended[:, :, 1:, 1:] = mask
  return mask_extended    


def get_mask_attn_dilated(q, k):

  q_adjusted = q[:, :, 1:, :]
  k_adjusted = k[:, :, 1:, :]
  B, H, N, D = q_adjusted.size()
  w = N // 1
  mask = torch.zeros((B, H, N, N), device=q.device)
  fib_set = set()
  a, b = 1, 1
  while a <= w:
    fib_set.add(a)
    fib_set.add(-a)
    a, b = b, a + b
  for i in fib_set:
    indices = torch.arange(max(-i, 0), min(N, N - i))
    mask[:, :, indices, indices + i] = 1
  mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device)
  mask_extended[:, :, 1:, 1:] = mask
  # print(mask_extended)
  # print(practical_mask_ratio(mask))
  return mask_extended  


def get_mask_attn_sampling_dilated(q, k):
  q_adjusted = q[:, :, 1:, :]
  k_adjusted = k[:, :, 1:, :]
  B, H, N, D = q_adjusted.size()
  mask_ratio = 0.2
  w = N // 4
  q_norms = torch.linalg.norm(q_adjusted, dim=-1, keepdim=True)
  k_norms = torch.linalg.norm(k_adjusted.transpose(-2, -1), dim=-2, keepdim=True)
  norms_matrix = q_norms * k_norms
  mask = torch.zeros((B, H, N, N), device=q.device, dtype=q.dtype)
  fib_set = set()
  a, b = 1, 1
  while a <= w:
    fib_set.add(a)
    fib_set.add(-a)
    a, b = b, a + b
  for offset in fib_set:
        diagonal = torch.diagonal(norms_matrix, offset=offset, dim1=-2, dim2=-1)
        topk = int(diagonal.size(-1) * (1 - mask_ratio))
        _, indices = torch.topk(diagonal, topk, dim=-1)
        diag_mask = torch.zeros_like(diagonal)
        diag_mask.scatter_(-1, indices, 1)
        torch.diagonal(mask, offset=offset, dim1=-2, dim2=-1).copy_(diag_mask)
  mask = (mask != 0).int()
  mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device, dtype=mask.dtype)
  mask_extended[:, :, 1:, 1:] = mask
  return mask_extended  

def get_mask_attn_shufle_dilated(q, k):
    
    q_adjusted = q[:, :, 1:, :]
    k_adjusted = k[:, :, 1:, :]
    B, H, N, _ = q_adjusted.size()
    w = N
    mask = torch.zeros((B, H, N, N), device=q.device)
    fib_sets = []
    for h in range(H):
        fib_set = set()
        a = 1 + h
        b = a * 2
        while a <= w:
            fib_set.add(a)
            fib_set.add(-a)
            a, b = b, a + b
        fib_sets.append(fib_set)
    random.shuffle(fib_sets)
    for h in range(H):
        fib_set = fib_sets[h]
        for i in fib_set:
            indices = torch.arange(max(-i, 0), min(N, N - i))
            mask[:, h, indices, indices + i] = 1
    mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device)
    mask_extended[:, :, 1:, 1:] = mask
    # print(practical_mask_ratio(mask))
    return mask_extended  


def helper_shuffle(i, shuffled_array):
    random.seed(i)
    random.shuffle(shuffled_array)
    return shuffled_array

def get_mask_attn_shufle_dilated_each_layer(q, epoch_or_depth_id):
    B, H, N, _ = q.shape
    N = N-1
    mask = torch.zeros((B, H, N, N), device=q.device, dtype=q.dtype)
    w = N // 3
    fib_sets = []
    for h in range(H):
        fib_set = []
        a = h + 1
        b = 2 * a
        while a <= w:
            fib_set.append(a)
            fib_set.append(-a)
            a, b = b, a + b
        fib_sets.append(fib_set)
    fib_sets = helper_shuffle(epoch_or_depth_id,fib_sets)
    for h in range(H):
        fib_set = fib_sets[h]
        for i in fib_set:
            indices = torch.arange(max(-i, 0), min(N, N - i), device=q.device)
            mask[:, h, indices, indices + i] = 1
    mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device, dtype=mask.dtype)
    mask_extended[:, :, 1:, 1:] = mask
    return mask_extended

def helper_shuffle(i, shuffled_array):
    random.seed(i)
    random.shuffle(shuffled_array)
    return shuffled_array

def generate_head_indices(N, h, omin,modified_flag=False):
    def get_fibonacci(a, b, w):
        sequence = [a]
        if b <= w:
            sequence.append(b)
        else:
            return sequence
        while True:
            new_element = sequence[-1] + sequence[-2]
            if new_element > w:
                break
            sequence.append(new_element)
        return sequence
    headindices = [[] for _ in range(h)]
    wmax = N // 3
    phi = (1 + math.sqrt(5)) / 2
    for i in range(1, h + 1):
        a = int(math.floor(math.floor(i * phi) * phi))
        b = int(math.floor(math.floor(i * phi) * phi ** 2))
        w = omin + int((wmax - omin) / (h - 1) * (i - 1))
        headindices[i - 1] = get_fibonacci(a, b, w)
        if modified_flag:
            if i>1:
                headindices[i - 1].insert(0,a-(i-1))
                headindices[i - 1].insert(0,i-1)
    headindices = [torch.tensor(seq, dtype=torch.int64) for seq in headindices]
    return headindices


def get_mask_attn_wythoff(q,depth_id,modified_flag=False):
    q_adjusted = q[:, :, 1:, :]
    B, H, N, _ = q_adjusted.size()
    headindices = generate_head_indices(N, H, 5,modified_flag)
    mask = torch.zeros((B, H, N, N), device=q.device, dtype=q.dtype)
    headindices = helper_shuffle(depth_id,headindices)
    for h in range(H):
        fib_indices = headindices[h]
        for i in fib_indices:
            indices = torch.arange(max(-i, 0), min(N, N - i))
            mask[:, h, indices, indices + i] = 1
            indices = torch.arange(max(i, 0), min(N, N + i))
            mask[:, h, indices, indices - i] = 1
        # print(f'h= {h}, fib_indices= {fib_indices}')
    mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device, dtype=mask.dtype)
    mask_extended[:, :, 1:, 1:] = mask
    nonzero_count = torch.count_nonzero(mask[0])
    total = N*N*H
    print(f'exp:mask_count: {total-nonzero_count}')
    print(f'exp:maskt percent: {(total-nonzero_count)/total}')
    return mask_extended    