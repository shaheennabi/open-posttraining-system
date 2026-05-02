
import torch
import logging
from logger import logging

## rope

def compute_rope_angles(head_dim, theta_base=10000, context_length=2048, dtype=torch.float32):
    
    assert head_dim % 2 == 0, "head_dim must be even"

    index = torch.arange(0, head_dim, 2, dtype=dtype)
    inv_freq = 1.0 / (theta_base ** (2 * index / head_dim))

    ## compute positions
    positions = torch.arange(context_length, dtype=dtype) ## we are calculating the positions here  

    ## computing the angle 
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) ### positions = 2048, 1  ** 1 ---- inv_freq = head_dim // 2 -----> angles = 2048 * 64

   
    cos = torch.cos(angles) ## 2048, 64
    sin = torch.sin(angles)  ## 2048, 64

    return  cos, sin


def apply_rope(x, cos, sin):

    B, T, H, D = x.shape

    x1 = x[..., ::2]   # [B, T, H, D/2]
    x2 = x[..., 1::2]  # [B, T, H, D/2]

    # Handle both training and inference
    if cos.dim() == 2:  # training
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D/2]
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:               # inference (single position)
        cos = cos[None, None, None, :]       # [1, 1, 1, D/2]
        sin = sin[None, None, None, :]

    x_even = x1 * cos - x2 * sin
    x_odd  = x1 * sin + x2 * cos

    x_out = torch.stack([x_even, x_odd], dim=-1).flatten(-2) # (b, t, h, d/2, 2) --> # (b, t, h, d)

    return x_out  # (b, t, h, d)


## RMSNorm
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, emb, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb))

    def forward(self, x):  ## x = (b,t,emb_dim)
        square = torch.square(x)
        sq_mean = square.mean(dim=-1, keepdim=True)  #b,t,1
        value = sq_mean +  self.eps
        rms_value = torch.sqrt(value)
        normalized_value = x / rms_value
        value = normalized_value * self.weight  ## b,t,d -- ## b, t, 1 
        return value   # b, t, d


    
## group-query attention

import torch
import torch.nn as nn

class GroupQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, head_dim, kv_heads, dtype=torch.float32) :
        super().__init__()

        self.d_in = d_in
        self.num_heads = num_heads
        self.h_dim = head_dim    ## dimension per head
        self.kv_heads = kv_heads
        self.group_size = num_heads // kv_heads
        self.d_out = num_heads * self.h_dim


        self.w_query = nn.Linear(self.d_in, self.d_out,  dtype=dtype)  ## (b, t, num_heads * h_dim)
        self.w_keys = nn.Linear(self.d_in, self.kv_heads * self.h_dim, dtype=dtype)    ## (b, t, kv_heads * h_dim) 
        self.w_values = nn.Linear(self.d_in, self.kv_heads * self.h_dim, dtype=dtype)  ## (b, t, kv_heads * h_dim)

        self.proj_out = nn.Linear(self.d_out, self.d_in, dtype=dtype)   ## (b, t, num_heads * h_dim)
        

        self.q_norm = RMSNorm(self.d_out) ## (b, t, num_heads * h_dim)
        self.k_norm = RMSNorm(self.kv_heads * self.h_dim)  ##  (b, t, kv_heads * h_dim)
        






    def forward(self, x, cos, sin, mask, cache=None):
        b, t, _ = x.shape

        logging.debug("GQA forward b=%s t=%s cache=%s", b, t, cache is not None)
        query = self.w_query(x)   ## x = (b, t, d_in)   -->  (b, t, num_heads * h_dim)
        keys = self.w_keys(x)    ## x = (b, t, d_in)  --> (b, t, kv_heads * h_dim)
        values = self.w_values(x)   ## x = (b, t, d_in) --> (b, t, kv_heads * h_dim)

        ## reshaping
        query = query.view(b, t, self.num_heads, self.h_dim)  ## (b, t, num_heads, h_dim)
        keys_new = keys.view(b, t, self.kv_heads, self.h_dim)    ## (b, t, kv_heads, h_dim)
        values_new = values.view(b, t, self.kv_heads, self.h_dim)   ## (b, t, kv_heads, h_dim)

        ## rope
        query = apply_rope(query, cos, sin) ##  rope expects = (b, t, num_heads, d)
        keys_new = apply_rope(keys_new, cos, sin)   ## rope expects = (b, t, kv_heads, d)

        ## reshaping --for kv cache
        query = query.transpose(1, 2)  ## (b, num_heads, t, d)
        keys_new = keys_new.transpose(1,2)  ## (b, kv_heads, t, d)
        values_new = values_new.transpose(1,2)  ## (b, kv_heads, t, d)

        ## cache 
        ## expects ---> (b, kv_heads, t, d)
        if cache is not None: 
            prev_k, prev_v = cache 
            keys = torch.cat([prev_k, keys_new], dim=2)  ## keys_new.shape == (b, kv_heads, t, d), prev_k.shape == (b, kv_heads, t, d ) ----> keys_new = (b,kv_heads, prev_k + t, d)
            values = torch.cat([prev_v, values_new], dim=2)   ## values_new.shape == (b, kv_heads, t, d), prev_v.shape == (b, kv_heads, t, d) ----> keys_new = (b, kv_heads, prev_v + t, d)

        else: 
            keys, values = keys_new, values_new  ## (b, kv_heads, t, d)
        next_cache = (keys, values)   ## tuple((b, kv_heads, t, d), (b, kv_heads, t, d))

        ## getting back the num_heads shape... 
        keys = torch.repeat_interleave(keys, self.group_size, dim=1)  ## (b, num_heads, t, d)
        values = torch.repeat_interleave(values, self.group_size, dim=1)  ## (b, num_heads, t, d)


        ## attention
        attn_scores = query @ keys.transpose(2, 3)  ## query = (b, num_heads, t, d) --- keys = (b, num_heads, d, t)  ---> attn_scores = (b, num_heads, t, t)

        ## scale logits
        attn_scores = attn_scores / (self.h_dim ** 0.5)   ## (b, num_heads, t, t)

        ## apply mask
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)   ## (b, num_heads, t, t)
 
        ## softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)  ## (b, num_heads, t, t) ## taking softmax --horizontally over keys or column (like from left to right)

        ## applying attention to values
        context = attn_weights @ values   ## attn_weights = (b, num_heads, t, t) --- values = (b, num_heads, t, d) ---> context = (b, num_heads, t, d)

        ## merge heads
        context = context.transpose(1, 2) ## (b, t, num_heads, d)

        context = context.reshape(b, t, self.d_out)   ## (b, t, num_heads * d)
        context = self.proj_out(context)   ## (b, t, D_model)

        return context, next_cache   ## (b, t, D_model), cache tuple


        

import torch
import torch.nn as nn 
import torch.nn.functional as F  


class FeedForward(nn.Module):
    def  __init__(self, cfg):
        super().__init__()

        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)  ## (b,t, d_model) ---> (b,t, d_ff)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)  ## (b,t, d_model)  ---> (b,t, d_ff)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)  ## (b,t, d_ff) ---> (b,t, d_model)

    def forward(self, x):
        # x = (b, t, emb_dim) or (batch, seq_len, d_model)
        x_fc1 = self.fc1(x)  ## (b, t, emb_dim) --> (b, t, d_ff)
        x_fc2 = self.fc2(x)  ## (b,t, emb_dim) --> (b,t, d_ff)
        x_fc3 = self.fc3(x)  ## (b, t, d_ff)  --> (b,t, emb_dim)

        ## swiglu activation
        x = F.silu(x_fc1) * x_fc2  ## (b,t, d_ff) * (b,t, d_ff) --> (b,t, d_ff)
        
        ## projection back 
        out = self.fc3(x)  ## (b,t, d_ff) ----> (b,t, emb_dim)

        return out  ## (b,t, emb_dim)




## transformer block 
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        ## attention
        self.att = GroupQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],            
            kv_heads=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
            )
        
        self.ff = FeedForward(cfg)  ## (b,t, emb_dim)
        ## norm (b,t,D) -->  (b,t,D) ... same for both 
        self.rms_norm1 = RMSNorm(cfg["emb_dim"])
        self.rms_norm2 = RMSNorm(cfg["emb_dim"])


    def forward(self, x, mask, cos, sin, cache=None):
        ## x = (b,t,d_model)
        shortcut = x
        x = self.rms_norm1(x)  ## (b,t,D)
        x, next_cache = self.att(x, cos, sin, mask, cache=cache)  ##   (b,t, emb_size)
        x = x + shortcut

        ## shortcut connection for feed-forward block
        shortcut = x ### (b,t, emb_size)
        x = self.ff(x)  ## (b,t, emb_size)
        x = self.rms_norm2(x)  ## (b,t, emb_size)
        ## residual
        x = x + shortcut  ## (b,t, emb_size)

        return x, next_cache
         ### (b, t, emb_size ) and Kv tensors 



## this is the block here ...
import torch
import torch.nn as nn

class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])  ## input = (b, t)  ---> (b,t,emb_dim)
        self.t_block = nn.ModuleList(
            TransformerBlock(cfg) for _  in range(cfg["n_layers"])  ### (b,t, emb_dim)
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])  ## (b,t,emb_dim)  --->  (b,t,emb_dim)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]) ## (b,t, emb) --->  (b,t, vocab_size)
    

        self.cfg = cfg
        logging.info("Initializing Qwen3Model vocab_size=%s emb_dim=%s n_layers=%s", cfg["vocab_size"], cfg["emb_dim"], cfg["n_layers"])

        if cfg["head_dim"] is None: 
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        
        else:
            head_dim = cfg["head_dim"]


        ## cos, and sin ---for rope
        cos, sin = compute_rope_angles(head_dim=head_dim, theta_base=cfg["rope_base"], context_length=cfg["context_length"])


        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.current_position = 0

    
    def forward(self, in_ids, cache=None):
        logging.info("Qwen3Model forward start input_ids=%s cache=%s", in_ids.shape, cache is not None)
        token_emb = self.emb(in_ids)
        x = token_emb   ##(b,t,emb_dim)

        num_tokens = x.shape(1)

        if cache is not None:
            ## inference mode 
            start_pos = self.current_position
            end_pos = start_pos + num_tokens
            self.current_position =  end_pos

            ## create the masks
            mask = torch.triu(
                torch.ones(end_pos, end_pos, device=x.device, dtype=torch.bool), diagonal=1,
            )[start_pos:end_pos, :end_pos]
            # rows:   start_pos → end_pos   → num_tokens
            # cols:   0 → end_pos           → total tokens so far
            # mask.shape = (num_tokens, end_pos)

        else: ## training mode 
            start_pos = 0 
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
                # (num_tokens, num_tokens)

        ### broadcast mask
        mask = mask[None, None, :, :]  ## shape = (1,1,num_tokens,num_tokens) or (1,1,new_tokens, end_pos)

        for i, block in enumerate(self.t_block):
            blk_cache = cache.get(i) if cache else None

            ## shape = (b, t, emb_dim)
            x, new_blk_cache = block(x, mask, self.cos, self.sin, cache=blk_cache)

            if cache is not None: 
                cache.update(i, new_blk_cache)

        x = self.final_norm(x) ## shape = (b,t,emb_dim)

        logits = self.out_head(x.to(self.cfg["dtype"]))  ### (b,t,emb_dim)  ---> (b,t, vocab_size)
        logging.info("Qwen3Model forward end logits=%s", logits.shape)
        return logits 
    
    def reset_kv_cache(self):
        self.current_position = 0  ## must be called, when starting a new independent sequence
        logging.info("KV cache reset")
    


## kv cache 
class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]
    
    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache
    
    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None




import re
from tokenizers import Tokenizer
from pathlib import Path

class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]

    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)") ### finds the special tokens inside text

    def __init__(self, tokenizer_file_path="tokenizer.json",
                    apply_chat_template=False,
                    add_generation_prompt=False,
                    add_thinking=False):        
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt =  add_generation_prompt
        self.add_thinking = add_thinking


        tok_path = Path(tokenizer_file_path)   ## this is to have, the path of the tokenizer file.
        if not tok_path.is_file():
            raise FileNotFoundError(f"The {tok_path} is not found") 
        
        self._tok = Tokenizer.from_file(str(tok_path)) ## loading the tokenizer file--from it's location using Tokenizer from huggingface tokenizers 
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}   ## iterating over the _SPECIAL, tokens and storing their ID's in {key:pair} format.
        self.pad_token = "<|endoftoken|>"   ### this is pad_token
        self.pad_token_id = self._special_to_id.get(self.pad_token)  ## getting the pad_token ID

        ## match hf behaviour: chat_model --> <|im_end|>, base_model --> <|endoftext|> 
        fname = tok_path.name.lower()   ## lowering the path name
        if "base" in fname and "reasoning" not in fname:  ## if it's a "base" model and not "reasoning" modlel then it's "end of text"
            self.eos_token = "<|endoftext|>"
        else: 
            self.eos_token = "<|im_end|>"     ### otherwise "im end" 
        self.eos_token_id = self._special_to_id.get(self.eos_token)  ## getting the precomputed token id



    ## back here tomorrow, again
    def encode(self, prompt, chat_wrapped=None): 
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template  
        


        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:  ## checking if it's a special token--directly get the id
            return [self._special_to_id[stripped]]
     


        if chat_wrapped:
            prompt = self._wrap_chat(prompt)  ##  if the condititon is false,(mostly during --training on raw text), this won't execute.


        Ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):  ## splliting the text--keepking special tokens as it's and splitting rest and remove empty strings etc.
            if part in self._SPECIALS:
                Ids.append(self._special_to_id[part]) ##  appending each item (special token's already computed Ids)
            else:
                Ids.extend(self._tok.encode(part).ids) ## extending (multiple ids at ones, to avoid nested append.)
        return  Ids
    

    def decode(self, token_ids):
        return self._tok.decode(token_ids, skip_special_tokens=False)    ## decoding Id's--back to text and also keeping the special specials...

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking: ## true
                s += "\n" ##  no thinking tags,,,  only new line..and let model generate freely
            else: ##false
                s += "\n<think>\n\n</think>\n\n"   ## add the thinking tags--and force the model to think inside tags... etc

        return s



## ok, load hf weights into qwen
import  torch
def load_hf_weights_into_qwen(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"There is a shape mismatch in tensor {tensor_name}. Left Shape: {left.shape}, Right Shape {right.shape}")
        
        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right) ## copy the weights, into left
            else: 
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))   ## convert format to tensors and copy to left
        
        return left
    

    model.emb.weight = assign(model.emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):
        block = model.t_block[l]
        att = block.att
        
        ## these are the q,k,v projections
        att.w_query.weight = assign(att.w_query.weight, params[f"model.layers.{l}.self_attn.q_proj.weight"], f"model.layers.{l}.self_attn.q_proj.weight")
        att.w_keys.weight = assign(att.w_keys.weight, params[f"model.layers.{l}.self_attn.k_proj.weight"], f"model.layers.{l}.self_attn.k_proj.weight")
        att.w_values.weight = assign(att.w_values.weight, params[f"model.layers.{l}.self_attn.v_proj.weight"], f"model.layers.{l}.self_attn.v_proj.weight")

        ### output projection
        att.proj_out.weight = assign(att.proj_out.weight, params[f"model.layers.{l}.self_attn.o_proj.weight"], f"model.layers.{l}.self_attn.o_proj.weight")



        ## q,k norm
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.weight = assign(att.q_norm.weight, params[f"model.layers.{l}.self_attn.q_norm.weight"], f"model.layers.{l}.self_attn.q_norm.weight")

        if hasattr(att, "k_norm")  and att.k_norm is not None:
            att.k_norm.weight = assign(att.k_norm.weight, params[f"model.layers.{l}.self_attn.k_norm.weight"], f"model.layers.{l}.self_attn.k_norm.weight")
                   
        if hasattr(block, "rms_norm1"):
            block.rms_norm1.weight = assign(block.rms_norm1.weight, params[f"model.layers.{l}.input_layernorm.weight"], f"model.layers.{l}.input_layernorm.weight")


        ## Feedforward weightss
        if "num_experts" in param_config:
            ## load router (gating weights)
            block.ff.gate.weight = assign(block.ff.gate.weight, params[f"model.layers{l}.mlp.gate.weight"], f"model.layers{l}.mlp.gate.weight")


            ## load expert weights
            for e in range(param_config["n_layers"]):

                prefix = f"model.layers{l}.mlp.experts.{e}"

                block.ff.fc1[e].weight = assign(block.ff.fc1[e].weight, params[f"{prefix}.gate_proj.weight"], f"{prefix}.gate_proj.weight")

                block.ff.fc2[e].weight = assign(block.ff.fc2[e].weight, params[f"{prefix}.up_proj.weight"], f"{prefix}.up_proj.weight")

                block.ff.fc3[e].weight = assign(block.ff.fc3[e].weight, params[f"{prefix}.down_proj.weight"], f"{prefix}.down_proj.weight")


                ## moving layers to cpu
                block.ff.fc1[e] = block.ff.fc1[e].to("cpu")
                block.ff.fc2[e] = block.ff.fc2[e].to("cpu")
                block.ff.fc3[e] = block.ff.fc3[e].to("cpu")
        
        else:

            block.ff.fc1.weight = assign(block.ff.fc1.weight, params[f"model.layers{l}.mlp.gate_proj.weight"], f"model.layers{l}.mlp.gate_proj.weight")

            block.ff.fc2.weight = assign(block.ff.fc2.weight, params[f"model.layers{l}.mlp.up_proj.weight"], f"model.layers{l}.mlp.up_proj.weight")

            block.ff.fc3.weight = assign(block.ff.fc3.weight, params[f"model.layers{l}.mlp.down_proj.weight"], f"model.layers{l}.mlp.down_proj.weight")



        if hasattr(block, "rms_norm2"):
            block.rms_norm2.weight = assign(block.rms_norm2.weight, params[f"model.layers.{l}.post_attention_layernorm.weight"], f"model.layers.{l}.post_attention_layernorm.weight")

    ## final norm and output head
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")


    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = model.emb.weight
        logging.info("Weight tying enabled: output head shares embedding weights")









# 0.6 billion parameters ## copied from sebastian raschka's notebook...
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,     # Vocabulary size
    "context_length": 40_960,  # Length originally used during training
    "emb_dim": 1024,           # Embedding dimension
    "n_heads": 16,             # Number of attention heads
    "n_layers": 28,            # Number of layers
    "hidden_dim": 3072,        # Size of intermediate dim in FeedForward
    "head_dim": 128,           # Size of the heads in GQA
    "qk_norm": True,           # Whether to normalize queries & keys in GQA
    "n_kv_groups": 8,          # Key-Value groups for GQA
    "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,   # Lower-precision dtype to reduce memory
}