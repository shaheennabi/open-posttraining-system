
import sys
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from logger import logging
import torch.nn as nn
import torch.nn.functional as F  
import re
from tokenizers import Tokenizer




## rope

def compute_rope_angles(head_dim, theta_base=10000, context_length=2048, dtype=torch.float32):
    
    assert head_dim % 2 == 0, "head_dim must be even"

    index = torch.arange(0, head_dim, 2, dtype=dtype)
    inv_freq = 1.0 / (theta_base ** (index / head_dim))

    ## compute positions
    positions = torch.arange(context_length, dtype=dtype) ## we are calculating the positions here  

    ## computing the angle 
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) ### positions = 2048, 1  ** 1 ---- inv_freq = head_dim // 2 -----> angles = 2048 * 64

   
    cos = torch.cos(angles) ## 2048, 64
    sin = torch.sin(angles)  ## 2048, 64

    return  cos, sin


def apply_rope(x, cos, sin):

    B, T, H, D = x.shape

    x1 = x[..., : D // 2]   # [B, T, H, D/2]
    x2 = x[..., D // 2 :]   # [B, T, H, D/2]

    # Handle both training and inference
    if cos.dim() == 2:  # training
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D/2]
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:               # inference (single position)
        cos = cos[None, None, None, :]       # [1, 1, 1, D/2]
        sin = sin[None, None, None, :]

    x_first = x1 * cos - x2 * sin
    x_second = x2 * cos + x1 * sin

    return torch.cat([x_first, x_second], dim=-1)  # (b, t, h, d)


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


        self.w_query = nn.Linear(self.d_in, self.d_out,  dtype=dtype, bias=False)  ## (b, t, num_heads * h_dim)
        self.w_keys = nn.Linear(self.d_in, self.kv_heads * self.h_dim, dtype=dtype, bias=False)    ## (b, t, kv_heads * h_dim) 
        self.w_values = nn.Linear(self.d_in, self.kv_heads * self.h_dim, dtype=dtype, bias=False)  ## (b, t, kv_heads * h_dim)

        self.proj_out = nn.Linear(self.d_out, self.d_in, dtype=dtype, bias=False)   ## (b, t, num_heads * h_dim)
        

        self.q_norm = RMSNorm(self.h_dim) ## Normalize per head dimension
        self.k_norm = RMSNorm(self.h_dim)  ## Normalize per head dimension
        






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

        query = self.q_norm(query)
        keys_new = self.k_norm(keys_new)

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

        ## swiglu activation
        x = F.silu(x_fc1) * x_fc2  ## (b,t, d_ff) * (b,t, d_ff) --> (b,t, d_ff)
        
        ## projection back 
        out = self.fc3(x)  ## (b,t, d_ff) ----> (b,t, emb_dim)

        return out  ## (b,t, emb_dim)




## transformer block 

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
        x = self.rms_norm2(x)  ## (b,t, emb_size)
        x = self.ff(x)  ## (b,t, emb_size)
        ## residual
        x = x + shortcut  ## (b,t, emb_size)

        return x, next_cache
         ### (b, t, emb_size ) and Kv tensors 






class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # ---------------------------------------------------------
        # Main model parameters
        # ---------------------------------------------------------

        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"],
            dtype=cfg["dtype"]
        )

        self.trf_blocks = nn.ModuleList(
            [
                TransformerBlock(cfg)
                for _ in range(cfg["n_layers"])
            ]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False,
            dtype=cfg["dtype"]
        )


        # ---------------------------------------------------------
        # Reusable RoPE utilities
        # ---------------------------------------------------------

        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]


        cos, sin = compute_rope_angles(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )


        ## register cos/sin as buffers
        ## these move automatically with model.to(device)
        ## but are not trainable parameters
        self.register_buffer(
            "cos",
            cos,
            persistent=False
        )

        self.register_buffer(
            "sin",
            sin,
            persistent=False
        )


        self.cfg = cfg

        ## track current position when using KV cache
        self.current_pos = 0



    def forward(self, in_idx, cache=None):

        # TOKEN EMBEDDINGS

        ## in_idx shape --> (b, t)

        tok_embeds = self.tok_emb(in_idx)

        ## tok_embeds shape --> (b, t, emb_dim)

        x = tok_embeds


        ## number of tokens in current forward pass
        num_tokens = x.shape[1]



        # BUILD CAUSAL ATTENTION MASK

        if cache is not None:

            ## used during autoregressive generation with KV cache

            pos_start = self.current_pos
            pos_end = pos_start + num_tokens

            self.current_pos = pos_end


            query_positions = torch.arange(pos_start, pos_end, device=x.device)


            key_positions = torch.arange(pos_end, device=x.device)


            mask = (key_positions[None, :] > query_positions[:, None])


        else:

            ## for our interpretability experiment
            ## we pass the complete prompt at once
            ## therefore pos_start = 0

            pos_start = 0


            ## standard causal mask
            ##
            ## token i cannot attend to future tokens j > i

            mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)


        ## broadcast causal mask
        ##
        ## (t, t)
        ##    ↓
        ## (1, 1, t, t)

        mask = mask[None, None, :, :]



        # SLICE RoPE VALUES FOR CURRENT POSITIONS
        cos = self.cos[pos_start : pos_start + num_tokens].to(device=x.device, dtype=x.dtype)


        sin = self.sin[pos_start : pos_start + num_tokens].to(device=x.device, dtype=x.dtype)



        # INTERPRETABILITY STORAGE

        ## we will store ONE vector from every transformer layer:
        ##
        ## the hidden representation of the LAST prompt token
        ##
        ## each stored vector has shape:
        ##
        ## (b, emb_dim)

        hidden_layers = []



        # TRANSFORMER BLOCKS

        for i, block in enumerate(self.trf_blocks):

            blk_cache = (cache.get(i) if cache is not None else None)


            ## before transformer block:
            ##
            ## x shape --> (b, t, emb_dim)


            x, new_blk_cache = block(x, mask, cos, sin, cache=blk_cache)


            ## after transformer block:
            ##
            ## x shape --> (b, t, emb_dim)
            ##
            ## x contains the representation of EVERY token
            ## after transformer layer i


            # INTERPRETABILITY EXTRACTION
            ## extract ONLY the last prompt-token representation
            ##
            ## x:
            ##     (b, t, emb_dim)
            ##
            ## x[:, -1, :]:
            ##     (b, emb_dim)
            ##
            ## because we process one complete prompt without
            ## padding, index -1 is the final prompt token

            last_token_hidden_layer = x[:, -1, :]


            ## store this layer's last-token representation

            hidden_layers.append(last_token_hidden_layer)


            ## update KV cache only when cache is being used

            if cache is not None:cache.update(i, new_blk_cache)



        # FINAL NORMALIZATION

        ## IMPORTANT:
        ##
        ## hidden_layers were collected BEFORE this final RMSNorm.
        ##
        ## Therefore our representations correspond to:
        ##
        ## post-transformer-block
        ## pre-final-RMSNorm
        ##
        ## residual-stream representations.


        ## x shape remains:
        ##
        ## (b, t, emb_dim)

        x = self.final_norm(x)



        # LANGUAGE MODEL HEAD

        ## project:
        ##
        ## (b, t, emb_dim)
        ##        ↓
        ## (b, t, vocab_size)

        logits = self.out_head(x.to(self.cfg["dtype"]))



        # STACK HIDDEN REPRESENTATIONS

        ## before stacking:
        ##
        ## hidden_layers = [
        ##
        ##     layer_1 --> (b, emb_dim),
        ##     layer_2 --> (b, emb_dim),
        ##     layer_3 --> (b, emb_dim),
        ##     ...
        ##     layer_L --> (b, emb_dim)
        ##
        ## ]


        ## stack across transformer layers
        ##
        ## list of L × (b, emb_dim)
        ##
        ## becomes:
        ##
        ## (b, n_layers, emb_dim)

        hidden_layers = torch.stack(hidden_layers, dim=1)



        # RETURN

        ## logits:
        ##
        ## (b, t, vocab_size)
        ##
        ##
        ## hidden_layers:
        ##
        ## (b, n_layers, emb_dim)

        return logits, hidden_layers



    def reset_kv_cache(self):

        ## reset generation position
        self.current_pos = 0



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


def sample_next_token(logits, temperature=0.6, top_k=20, top_p=0.95):
    logits = logits.float()
    if temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        kth_values = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_values, -torch.inf)

    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_mask, -torch.inf)
        logits = torch.full_like(logits, -torch.inf)
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_text(model, tokenizer, prompt, max_new_tokens=128, temperature=0.6, top_k=20,
                  top_p=0.95, chat=True, enable_thinking=False):
    model.eval()
    if chat:
        input_ids = tokenizer.encode_chat(
            prompt,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    else:
        input_ids = tokenizer.encode(prompt)

    device = next(model.parameters()).device
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    cache = KVCache(model.cfg["n_layers"])
    model.reset_kv_cache()
    generated_ids = []

    with torch.inference_mode():
        logits = model(input_tensor, cache=cache)
        next_token = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        for _ in range(max_new_tokens):
            token_id = next_token.item()
            if token_id in tokenizer.stop_token_ids:
                break

            generated_ids.append(token_id)
            logits = model(next_token, cache=cache)
            next_token = sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

    return tokenizer.decode(generated_ids, skip_special_tokens=True)





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
        tokenizer_config = self._load_tokenizer_config(tok_path)
        self.pad_token = tokenizer_config.get("pad_token", "<|endoftext|>")
        self.pad_token_id = self._tok.token_to_id(self.pad_token)

        self.eos_token = tokenizer_config.get("eos_token", "<|im_end|>")
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        self.endoftext_token_id = self._tok.token_to_id("<|endoftext|>")
        self.stop_token_ids = {
            token_id for token_id in (self.eos_token_id, self.endoftext_token_id)
            if token_id is not None
        }



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
    

    def decode(self, token_ids, skip_special_tokens=False):
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)    ## decoding Id's--back to text

    def encode_chat(self, user_msg, add_generation_prompt=True, enable_thinking=False):
        prompt = self._wrap_chat(
            user_msg,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        return self.encode(prompt, chat_wrapped=False)

    def _load_tokenizer_config(self, tok_path):
        config_path = tok_path.with_name("tokenizer_config.json")
        if not config_path.is_file():
            return {}
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _wrap_chat(self, user_msg, add_generation_prompt=None, enable_thinking=None):
        if add_generation_prompt is None:
            add_generation_prompt = self.add_generation_prompt
        if enable_thinking is None:
            enable_thinking = self.add_thinking

        s = (
            f"<|im_start|>user\n"
            f"{user_msg}\n"
            f"<|im_end|>\n"
        )
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
            if not enable_thinking:
                s += "<think>\n\n</think>\n\n"

        return s



import torch
import logging
from pathlib import Path
from safetensors.torch import load_file

def load_hf_weights_into_qwen(model, param_config, params):
    """
    Load HuggingFace model weights into Qwen3Model.
    Supports both model variants (tok_emb/trf_blocks and emb/t_block).
    """
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    # Handle both model attribute naming conventions
    emb_layer = model.emb if hasattr(model, 'emb') else model.tok_emb
    blocks_layer = model.t_block if hasattr(model, 't_block') else model.trf_blocks
    
    emb_layer.weight = assign(emb_layer.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):  # noqa: E741
        block = blocks_layer[l]
        att = block.att

        # Q, K, V projections
        att.w_query.weight = assign(
            att.w_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.w_keys.weight = assign(
            att.w_keys.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.w_values.weight = assign(
            att.w_values.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # Output projection
        att.proj_out.weight = assign(
            att.proj_out.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        # QK norms
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.weight = assign(
                att.q_norm.weight,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.weight = assign(
                att.k_norm.weight,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        # Attention layernorm
        block.rms_norm1.weight = assign(
            block.rms_norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Feedforward weights
        if "num_experts" in param_config:
            # Load router (gating) weights
            block.ff.gate.weight = assign(
                block.ff.gate.weight,
                params[f"model.layers.{l}.mlp.gate.weight"],
                f"model.layers.{l}.mlp.gate.weight"
            )
            # Load expert weights
            for e in range(param_config["num_experts"]):
                prefix = f"model.layers.{l}.mlp.experts.{e}"
                block.ff.fc1[e].weight = assign(
                    block.ff.fc1[e].weight,
                    params[f"{prefix}.gate_proj.weight"],
                    f"{prefix}.gate_proj.weight"
                )
                block.ff.fc2[e].weight = assign(
                    block.ff.fc2[e].weight,
                    params[f"{prefix}.up_proj.weight"],
                    f"{prefix}.up_proj.weight"
                )
                block.ff.fc3[e].weight = assign(
                    block.ff.fc3[e].weight,
                    params[f"{prefix}.down_proj.weight"],
                    f"{prefix}.down_proj.weight"
                )
                # After assigning weights, move the expert layers from meta to CPU
                block.ff.fc1[e] = block.ff.fc1[e].to("cpu")
                block.ff.fc2[e] = block.ff.fc2[e].to("cpu")
                block.ff.fc3[e] = block.ff.fc3[e].to("cpu")

        else:
            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight"
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight"
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight"
            )

        block.rms_norm2.weight = assign(
            block.rms_norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Final normalization and output head
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = emb_layer.weight
        print("Model uses weight tying.")



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
