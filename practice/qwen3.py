# Qwen3 is a 0.6B-parameter model.
# Before implementing the model, we will first calculate:
# 1. Total number of parameters.
# 2. Parameter memory footprint.
# 3. Gradient memory requirements.
# 4. Optimizer state memory.
# 5. Activation memory during the forward and backward passes.
# 6. Total GPU memory required for training.

# Understanding these memory and compute requirements will help us
# make informed architectural and implementation decisions.

import jax
import jax.numpy as jnp

def Embedding(input, embedding_weight):
    """
    Embedding shape: [151936, 1024] (vocab_size, emb_dim)
    Parameters: 151936 * 1024 = 155.58M parameters.
    Parameter memory (BF16): 151936 * 1024 * 2 = 296.75 MiB. # 2 for BF16 precision
    Activation shape: [B, T, 1024].
    For B=2, T=4: 2 * 4 * 1024 * 2 = 16 KiB.

    Here LM head shares embedding weights (weight tying),
    so parameters are counted only once.

    """
    token_embedding = embedding_weight[input] #input_shape: [B,T], embedding_weight_shape: [151936,1024] ==> token_embedding_shape: [B,T,1024]
    #we will get [1024] dimensional vector for every token in Batch
    return token_embedding #shape: [B,T,1024]




def transformer_layer(x, input_rms_norm_weight, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, o_proj_weight, post_attention_norm_weight, gate_proj_weight, up_proj_weight, down_proj_weight, layer_key_cache, layer_value_cache, cache_position, rope_theta):

    # getting the normalized_output
    input_rms_norm = RMSNorm(x, input_rms_norm_weight) # shape: [B,T,1024]

    #  getting attention_output
    attention_output, updated_key_cache, updated_value_cache = self_attention_block(
        input_rms_norm, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, o_proj_weight,
        layer_key_cache, layer_value_cache, cache_position, rope_theta,
    ) # layer caches: [B,C,8,128] each

    # residual connection
    residual_added = x + attention_output #shape: x=[B,T,1024] + attention_output=[B,T,1024] => residual_added = [B,T,1024]

    # post-attention_layer norm
    post_attention_norm_output = RMSNorm(x=residual_added, gamma=post_attention_norm_weight) #input_shape: [B,T,1024] --> output_shape: [B,T,1024]  #post_attention_norm.weight = [1024]

    # gated mlp
    feedforward_network_output = FeedForwardNetwork(post_attention_norm_output, gate_proj_weight, up_proj_weight, down_proj_weight) #shapes: [B,T,!024], [3072,1024], [3072,1024], [1024,3072] --->  output_shape: [B,T,1024]

    # mlp residual-connection
    transformer_layer_output = residual_added + feedforward_network_output  #shapes: [B,T,1024] + [B,T,1024] --> [B,T,1024]

    return transformer_layer_output, updated_key_cache, updated_value_cache #shape: [B,T,1024], #[B,C,8,128], #[B,C,8,128]



def self_attention_block(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, o_proj_weight, layer_key_cache, layer_value_cache, cache_position, rope_theta=10000): #q_weight = [2048,1024], k_weight = [1024,1024], v_weight = [1024,1024]
    """
        Projection:
        q_proj.weight = [2048, 1024] # output_features, input_features
        k_proj.weight = [1024, 1024]
        v_proj.weight = [1024, 1024]
        o_proj.weight = [1024, 2048]

        all in BF16
        -------
        Query projection params: 28 * 2048 * 1024 = 58,720,256
        Key projection params: 28 * 1024 * 1024 = 29,360,128
        Value projection params: 28 *  1024 * 1024 = 29,360,128
        total params: 117,440,512  -->  117.440 millions parameters
        -------
        Parameter Memory:
        117,440,512 * 2 (Bf16) = 234,881,024 bytes
        Converting them to MiB's => 234,881,024 / 1024^2  = 224 MiB
        -----
        Now o_proj.weight = [1024,2048]
        Parameter count:  28 * 1024 * 2048 = 58,720,256 parameters
        Parameter Memory: 117,440,512 / 1024^2 = 112 MiB
        ----
        Forward FLOPs:
        Q = 2BT * 28 * (2048 * 1024) = 2BT * 58,720,256
        K = 2BT * 28 * (1024 * 1024) = 2BT * 29,360,128
        V = 2BT * 28 * (1024 * 1024) = 2BT * 29,360,128
        o_proj = 2BT * 28 * (2048 * 1024) = 2BT * 58,720,256
        Combined forward FLOPs:
        F_forward = 2BT * 176,160,768 = 352,321,536 BT Flops
        -----
        Backward flops:
        Q backward = 2 * (2BT * 58,720,256) = 2BT * 117,440,512
        K backward = 2 * (2BT * 29,360,128) = 2BT * 58,720,256
        V backward = 2 * (2BT * 29,360,128) = 2BT * 58,720,256
        o_proj backward = 2 * (2BT * 58,720,256) = 2BT * 117,440,512
        Combined backward FLOPs:
        F_backward = 2BT * 352,321,536 = 704,643,072 BT Flops
        ------
        Total projection FLOPs:
        F_total = F_forward + F_backward = 1,056,964,608 BT Flops
        ---

        RMSNorm Parameter:
        input_layernorm.weight: [1024]
        post_attention_layernorm.weight: [1024]
        q_norm.weight: [128]
        k_norm.weight: [128]
        Input RMSNorm parameters: 28 * 1024 = 28,672
        Post-attention RMSNorm parameters: 28 * 1024 = 28,672
        Q-Norm parameters: 28 * 128 = 3,584
        K-Norm parameters: 28 * 128 = 3,584
        Final model RMSNorm: 1 * 1024 = 1,024
        Total RMSNorm parameters: 65,536
        Parameter memory (BF16):
        65,536 * 2 = 131,072 bytes
        131,072 / 1024^2 = 0.125 MiB
        -------
        RMSNorm FLOPs (rough estimates):
        Normalized dimensions per token:
        28 * (2 * 1024 + 16 * 128 + 8 * 128) + 1024 = 144,384
        F_RMS_forward ~= 4 * 144,384 * BT = 577,536 BT
        F_RMS_backward ~= 8 * 144,384 * BT = 1,155,072 BT
        -------

        RoPE:
        Q rotation pairs per layer: 16 * 64 = 1,024
        K rotation pairs per layer: 8 * 64 = 512
        Total rotation pairs per layer: 1,536
        Each pair requires approximately: 4 multiplications + 2 additions = 6 FLOPs
        Forward RoPE FLOPs: 6 * 1,536 * 28 * BT = 258,048 BT
        -------
        Updated Forward FLOPs:
        F_forward = Projection FLOPs + RMSNorm FLOPs + RoPE FLOPs
        = 352,321,536 BT + 577,536 BT + 258,048 BT = approximately 353,157,120 BT
        Updated Backward FLOPs:
        F_backward = Projection backward + RMSNorm backward + RoPE backward
        = 704,643,072 BT + 1,155,072 BT + 258,048 BT = approximately 706,056,192 BT
        ------
        Grouped-Query Attention:
        Each KV head is shared by 2 query heads.

        Attention Forward FLOPs:
        QK multiplication: F_QK = 2 * B * 16 * T^2 * 128 * 28 = 114,688 * B * T^2
        Attention-weighted Value:
        F_AV = 2 * B * 16 * T^2 * 128 * 28 = 114,688 * B * T^2
        Total attention forward FLOPs: F_attention = 229,376 * B * T^2
        Backward matrix multiplication FLOPs:
        F_backward ≈ 2 * F_attention = 458,752 * B * T^2

        -------
        Attention Activation Memory:
        Attention scores [B, 8, 2, T, T] (FP32): M_scores = 64 * B * T^2 bytes
        Attention probabilities [B, 8, 2, T, T] (BF16): M_probs = 32 * B * T^2 bytes
        Combined logical memory: M_attention = 96 * B * T^2 bytes
        Attention output [B, T, 16, 128] (BF16): M_output = 4,096 * B * T bytes
        -------
        GQA KV Cache Memory:
        Key and Value caches: Shape = [B, T, 8, 128] each
        Combined BF16 memory: M_KV = 2 * B * T * 8 * 128 * 2 = 4,096 * B * T bytes
        Equivalent MHA (16 KV heads): M_KV_MHA = 8,192 * B * T bytes
        GQA reduces KV cache memory by 50% compared
        with MHA using 16 KV heads.
        -------
        Summary:
        Forward attention FLOPs = 229,376 * B * T^2
        Backward attention FLOPs ≈ 458,752 * B * T^2
        Attention activation memory = 96 * B * T^2 bytes
        Attention output memory = 4,096 * B * T bytes
        KV cache memory = 4,096 * B * T bytes
        GQA adds zero parameters and reduces KV cache
        memory by 50% compared with equivalent MHA.
    """


    B,T,_ = x.shape
    # qkv projections
    query_proj =  x @ jnp.transpose(q_weight)  #XQ^T Shape: [B,T,2048]
    key_proj = x @ jnp.transpose(k_weight) #XK^T Shape: [B,T,1024]
    value_proj = x @ jnp.transpose(v_weight) #XV^T Shape: [B,T,1024]
    # reshaping
    query = query_proj.reshape(B,T,16, 128) #[B,T,16,128]
    key = key_proj.reshape(B,T,8,128) #[B,T,8,128]
    value = value_proj.reshape(B,T,8,128) #[B,T,8,128]


    # qk_norm operation
    q_norm = RMSNorm(query, q_norm_weight ,eps=1e-06) # output shape: [B,T,16,128]
    k_norm = RMSNorm(key, k_norm_weight, eps=1e-06) # output shape: [B,T,8,128]

    # apply rotary encoding (ROPE)
    # split both into halves
    q1, q2 = jnp.split(q_norm, 2, axis=-1) # shape: [B,T,16,64] + [B,T,16,64]
    k1, k2 = jnp.split(k_norm, 2, axis=-1) # shape: [B,T,8,64] + [B,T,8,64]

    # get the shape
    B,T,_, head_dim  = q_norm.shape
    # calculate the inverse frequency

    pairs = jnp.arange(0,head_dim, 2, dtype=jnp.float32)
    inv_freq = rope_theta ** (-pairs / head_dim) #shape: [64] # this is the inverse frequency, we will multiply with token positions to get their angles,

    # get the token positions
    positions = cache_position + jnp.arange(T, dtype=jnp.float32) #shape: [T]
    angle =  positions[:, None] * inv_freq[None, :] # shape: positions=[T,1] * inv_freq=[1,64] --> angle=[T,64]

    # now get cosine and sin for angle
    cos_angle = jnp.cos(angle) #shape: [T,64]
    sin_angle = jnp.sin(angle) #shape: [T,64]

    # now to apply these angle values to q,k
    # first get the dimension properly
    cos_angle = cos_angle[None, :, None, :] #shape: [1,T,1,64]
    sin_angle = sin_angle[None, :, None, :] #shape: [1,T,1,64]

    q1_prime = q1*cos_angle - q2*sin_angle #shape: [B,T,16,64]
    q2_prime = q2*cos_angle + q1*sin_angle #shape: [B,T,16,64]

    k1_prime = k1*cos_angle - k2*sin_angle #shape: [B,T,8,64]
    k2_prime = k2*cos_angle + k1*sin_angle #shape: [B,T,8,64]

    # now concat over, last_axis
    query_rotated = jnp.concatenate([q1_prime, q2_prime], axis=-1) #shape: [B,T,16,128]
    key_rotated = jnp.concatenate([k1_prime, k2_prime], axis=-1) #shape: [B,T,8,128]
    query_rotated = query_rotated.astype(jnp.bfloat16)  #Bf16 #Shape: [B,T,16,128]
    key_rotated = key_rotated.astype(jnp.bfloat16)  #Bf16 # Shape: [B,T,8,128]

    #kv-cache here.
    use_cache = layer_key_cache is not None
    if use_cache:
        indices = cache_position + jnp.arange(T) # absolute slots of the T new tokens, shape: [T]

        layer_key_cache = layer_key_cache.at[:, indices, :, :].set(key_rotated.astype(layer_key_cache.dtype))  # [B, C, 8, 128]
        layer_value_cache = layer_value_cache.at[:, indices, :, :].set(value.astype(layer_value_cache.dtype)) # [B, C, 8, 128]

        key_cached = layer_key_cache  #[B, C, 8, 128]
        value_cached = layer_value_cache  #[B, C, 8, 128]
    else:
        key_cached = key_rotated  #[B, T, 8, 128]  (no cache -> attend only to the current tokens)
        value_cached = value  #[B, T, 8, 128]


    #group-query attention -Self attention block
    num_query_heads = 16
    num_kv_heads = 8
    group_size = num_query_heads // num_kv_heads

    # reshape q into groups
    query_reshaped = query_rotated.reshape(B,T,num_kv_heads,group_size,head_dim) #shape: [B,T,8,2,128]
    query_for_scores = jnp.transpose(query_reshaped, axes=[0,2,3,1,4]) #[B,8,2,T,128]
    key_for_scores = jnp.transpose(key_cached, axes=[0,2,3,1]) #[B,8,128,C]
    key_for_scores = key_for_scores[:,:,None,:,:] #[B,8,128,T] --> [B,8,1,128,C]

    #dot product
    attention_scores = jnp.matmul(query_for_scores, key_for_scores, preferred_element_type=jnp.float32) / jnp.sqrt(jnp.float32(head_dim)) #shape: [B,8,2,T,C], fp32

    #causal_masking
    B,_,_,T,C = attention_scores.shape

    #query absolute positions
    query_positions = cache_position + jnp.arange(T) # cached_position + T tokens
    key_positions = jnp.arange(C) # C (slots not written yet are masked out here, since their position > every query position)
    mask = key_positions[None, :] <= query_positions[:, None]

    attention_scores = jnp.where(mask[None,None,None,:,:], attention_scores, -jnp.inf) #shape: [B,8,2,T,C]
    attention_scores = attention_scores.astype(jnp.float32)

    #softmax
    max_scores = jnp.max(attention_scores, axis=-1, keepdims=True)
    shifted_scores = attention_scores - max_scores
    exp_scores = jnp.exp(shifted_scores)
    sum_exp_scores = jnp.sum(exp_scores, axis=-1, keepdims=True)
    attention_probs = exp_scores / sum_exp_scores #shape: [B,8,2,T,C]
    attention_probs = attention_probs.astype(jnp.bfloat16) #shape: [B,8,2,T,C]

    #multiply with Value
    value_for_attention = value_cached[:,:,:,None,:] #shape: [B,C,8,128] -->  [B,C,8,1,128]
    value_for_attention = jnp.transpose(value_for_attention, axes=[0,2,3,1,4]) #shape: [B,8,1,C,128]

    attention_output = attention_probs @ value_for_attention #shape: [B,8,2,T,C] @ [B,8,1,C,128] = [B,8,2,T,128]

    attention_output = jnp.transpose(attention_output, axes=[0,3,1,2,4])  #shape: [B,T,8,2,128]
    output = attention_output.reshape(B,T,num_kv_heads*group_size,head_dim) #shape: [B,T,16,128]

    #output projection
    #concat the heads
    output =  output.reshape(B,T,num_query_heads*head_dim) #shape: [B,T,2048]
    final_output = output @ jnp.transpose(o_proj_weight) #shape: [B,T,2048] * [2048,1024] -> [B,T,1024]

    if not use_cache:
        return final_output, None, None
    return final_output, key_cached, value_cached   # [B,T,1024], # [B, C, 8, 128], # [B, C, 8, 128]




def RMSNorm(x, gamma, eps=1e-6):
    """
    Input activation shape:  [B, T, d_model]
    Output activation shape: [B, T, d_model]
    We have 28 transformer layers, each with an RMSNorm
    parameter vector (gamma) of size d_model = 1024.
    Total parameters: 28 * 1024 = 28,672 parameters.
    Parameter memory (BF16): 28 * 1024 * 2 = 57,344 bytes.
    RMSNorm operation:
    1. Square each element of the input.
    2. Calculate the mean of squared values.
    3. Add epsilon and take the square root.
    4. Divide the input by the calculated RMS value.
    5. Multiply by the learnable scaling parameter (gamma).

    forward flops are almost = 28 * B * T * 4D
    backward flops are almost = 28 * B * T * 8D
    """
    input_dtype = x.dtype
    x = x.astype(jnp.float32) # we have to get this into float32 precision to sustain the precision,
    temp_val = jnp.mean(jnp.square(x), axis=-1, keepdims=True) #eps added before,
    temp_val = temp_val + eps
    normed = x * jax.lax.rsqrt(temp_val) # x / sqrt(mean(x^2)+eps)
    normed = normed.astype(input_dtype)   # cast back first (this is what HF Qwen3RMSNorm does) ...
    return gamma * normed                 # ... then scale by the learnable weight. input_shape = output_shape




def FeedForwardNetwork(post_attention_norm_output, gate_proj_weight, up_proj_weight, down_proj_weight):
    """
    Parameters:
    Gate Projection:
    gate_proj = 3072 * 1024 * 28 = 88,080,384 parameters
    - parameter memory: 2 * 88,080,384 = 168 MiB
    - gate_projection forward flops: 2 * B * T * 3072 * 1024 * 28 = 176,160,768 BT
    - gate_projection backward flops: 352,321,536 BT
    ------
    Up Projection:
    up_proj = 3072 * 1024 * 28 = 88,080,384 parameters
    - parameter memory: 2 * 88,080,384 = 168 MiB
    - up_projection forward flops: 2 * B * T * 3072 * 1024 * 28 = 176,160,768 BT
    - up_projection backward flops: 352,321,536 BT
    -------
    SiLU Activation and SwiGLU Gating:
    Flops:
    Additions       = 86,016 BT
    Multiplications = 172,032 BT
    Exponential = 86,016 BT
    Division    = 86,016 BT
    F_SiLU_Gating = 86,016 + 172,032 + 86,016 + 86,016 = 430,080 Flops
    Activation Memory SwiGLU (BF16):
    M_total = 28 * 30,720 BT
    M_total = 860,160 BT bytes
    *feedforward_network_output Parameters:
    down_proj_weight = 1024 * 3072 * 28 = 88,080,384 parameters
    parameter Bf16 memory: 2 * 88,080,384 = 168 MiB memory
    Forward Flops: 28 * 2 * B * T * 3072 * 1024 =  176,160,768 BT
    Backward Flops: 352,321,536 BT
    Activation Memory: 6144BT +  2048BT = 8192 BT bytes per layer
    """

    # gate_proj
    gate_proj = post_attention_norm_output @ jnp.transpose(gate_proj_weight)   #post_attention_norm_output:shape: [B,T,1024] @  transpose(gate_projection_weight:shape: [3072,1024]) ---> [B,T,3072]

    up_proj = post_attention_norm_output @ jnp.transpose(up_proj_weight)  #post_attention_norm_output:shape: [B,T,1024] @ transpose(up_projection_weight:shape: [3072,1024]) --> [B,T,3072]

    silu_gate_proj = silu_activation(gate_proj)  #gate_proj_shape: [B,T,3072]

    #element-wise with up_proj
    h = silu_gate_proj * up_proj #shapes: [B,T,3072] * [B,T,3072] = [B,T,3072]

    #back to down_proj
    feedforward_network_output = h @ jnp.transpose(down_proj_weight)   #shapes:  [B,T,3072] @ [1024, 3072]^T => [B,T,1024]

    return feedforward_network_output #shape: [B,T,1024]



def silu_activation(x):

    sigmoid = 1 / (1 + jnp.exp(-x))  #x_shape: [B,T,3072]
    silu = x * sigmoid #x_shape: [B,T,3072] * sigmoid_shape: [B,T,3072] --->  silu_shape: [B,T,3072]
    return silu



# Main model — Qwen3 architecture

def qwen3_model(input_ids, params, key_cache, value_cache, cache_position, rope_theta): #key_cache and value_cache: shape: [28, B, C, 8, 128]
    """
        LM head = Weight: [151936, 1024] (transposed)
        Forward FLOPs:
            2 * B * T * 1024 * 151936
            = 311,164,928 * B * T
        BF16 logits activation memory:
            B * T * 151936 * 2 bytes = 303,872 * B * T bytes
        Training backward FLOPs:
            Approximately 2 * forward FLOPs.
        Embedding parameters are counted only once.
    """

    # embedding
    token_embedding  = Embedding(input_ids, params["embed_tokens"])   #shape: [B,T,1024]

    #save the token_embedding as ---hidden states -> it will be later overridden
    hidden_states = token_embedding #shape: [B,T,1024]

    # The layer weights are stacked along axis 0 ([28, ...]) so we can scan over them:
    # jax.lax.scan slices layer i out of every leaf of the pytree and compiles the layer body ONCE,
    # instead of unrolling 28 copies in Python.
    def layer_step(h, layer_inputs):
        lp, layer_key_cache, layer_value_cache = layer_inputs  # layer_*_cache: [B,C,8,128] (one layer sliced out of [28,B,C,8,128])
        h_new, updated_key_cache, updated_value_cache = transformer_layer(
            h,
            lp["input_layernorm"],
            lp["q_proj"],
            lp["k_proj"],
            lp["v_proj"],
            lp["q_norm"],
            lp["k_norm"],
            lp["o_proj"],
            lp["post_attention_layernorm"],
            lp["gate_proj"],
            lp["up_proj"],
            lp["down_proj"],
            layer_key_cache,
            layer_value_cache,
            cache_position,
            rope_theta,
        )  #shape: [B,T,1024]
        return h_new.astype(h.dtype), (updated_key_cache, updated_value_cache)

    hidden_states, (key_cache, value_cache) = jax.lax.scan(layer_step, hidden_states, (params["layers"], key_cache, value_cache))

    #final RMSNorm
    final_rms_norm = RMSNorm(hidden_states, params["final_norm"]) #[B,T,1024]

    # linear-layer  --->  lm_head
    logits = jnp.matmul(final_rms_norm, jnp.transpose(params["embed_tokens"]), preferred_element_type=jnp.float32) #[B,T,1024] @ [1024,151936] => [B,T,151936] (fp32 logits)

    return logits, key_cache, value_cache  # output_shape =  [B,T,151936], [B,C,8,128], [B,C,8,128]