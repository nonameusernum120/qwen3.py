#!/usr/bin/env python3

"""
Pure-Python runtime for qwen3.c checkpoints (Q8_0), minimal deps.

Dependencies: only Python 3 standard library and numpy.
"""

import argparse
import math
import mmap
import os
import struct
import sys
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Globals

GS = 0  # quantization group size


# ----------------------------------------------------------------------------
# Config & Weight structures (Pythonic)


@dataclass
class Config:
    magic_number: int
    version: int
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    head_dim: int
    shared_classifier: int
    group_size: int


@dataclass
class QuantizedTensor:
    q: np.ndarray  # int8 flat array
    s: np.ndarray  # float32 scales array (length = elements/GS)


@dataclass
class TransformerWeights:
    # fp32
    rms_att_weight: np.ndarray
    rms_ffn_weight: np.ndarray
    rms_final_weight: np.ndarray
    q_norm_weights: np.ndarray
    k_norm_weights: np.ndarray

    # quantized
    q_tokens: QuantizedTensor
    token_embedding_table: np.ndarray
    wq: List[QuantizedTensor]
    wk: List[QuantizedTensor]
    wv: List[QuantizedTensor]
    wo: List[QuantizedTensor]
    w1: List[QuantizedTensor]
    w2: List[QuantizedTensor]
    w3: List[QuantizedTensor]
    wcls: QuantizedTensor


class RunState:
    def __init__(self, cfg: Config):
        all_heads_dim = cfg.n_heads * cfg.head_dim
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.x = np.zeros(cfg.dim, dtype=np.float32)
        self.xb = np.zeros(all_heads_dim, dtype=np.float32)
        self.hb = np.zeros(cfg.hidden_dim, dtype=np.float32)
        self.hb2 = np.zeros(cfg.hidden_dim, dtype=np.float32)
        self.xq = QuantizedTensor(q=np.zeros(all_heads_dim, dtype=np.int8), s=np.zeros(all_heads_dim // GS, dtype=np.float32))
        self.hq = QuantizedTensor(q=np.zeros(cfg.hidden_dim, dtype=np.int8), s=np.zeros(cfg.hidden_dim // GS, dtype=np.float32))
        self.q = np.zeros(all_heads_dim, dtype=np.float32)
        self.att = np.zeros(cfg.n_heads * cfg.seq_len, dtype=np.float32)
        self.logits = np.zeros(cfg.vocab_size, dtype=np.float32)
        self.key_cache = np.zeros(cfg.n_layers * cfg.seq_len * kv_dim, dtype=np.float32)
        self.value_cache = np.zeros(cfg.n_layers * cfg.seq_len * kv_dim, dtype=np.float32)


class Transformer:
    def __init__(self):
        self.config: Optional[Config] = None
        self.weights: Optional[TransformerWeights] = None
        self.state: Optional[RunState] = None
        self._mm: Optional[mmap.mmap] = None
        self._file: Optional[object] = None


# ----------------------------------------------------------------------------
# Quantization helpers


def dequantize(qx: QuantizedTensor, out: np.ndarray):
    q = qx.q.astype(np.float32)
    s = qx.s
    scale = np.repeat(s, GS)
    np.multiply(q, scale, out=out)


def quantize(dst: QuantizedTensor, x: np.ndarray):
    # In-place quantization of x into dst
    n = x.shape[0]
    assert n % GS == 0
    x_view = x.reshape(-1, GS)
    wmax = np.max(np.abs(x_view), axis=1)
    scale = wmax / 127.0
    scale[scale == 0] = 1e-8
    dst.s[:] = scale.astype(np.float32)
    q = (x_view / scale[:, None]).round().clip(-127, 127).astype(np.int8)
    dst.q[:] = q.reshape(-1)


def q8_matmul(xout: np.ndarray, x: QuantizedTensor, w: QuantizedTensor, n: int, d: int):
    # W(d,n) @ x(n,) -> xout(d,), both inputs quantized
    # Compute in groups of GS, then accumulate with scales
    xq = x.q.astype(np.int32)
    xsg = x.s
    wq = w.q.astype(np.int32)
    wsg = w.s
    out = np.zeros(d, dtype=np.float32)
    for i in range(d):
        base = i * n
        acc = 0.0
        for j in range(0, n, GS):
            ival = np.int32(0)
            ival = np.dot(xq[j:j+GS], wq[base + j: base + j + GS])
            acc += float(ival) * wsg[(base + j)//GS] * xsg[j//GS]
        out[i] = acc
    xout[:] = out


# ----------------------------------------------------------------------------
# Checkpoint reader


def _read_config(header_bytes: bytes) -> Config:
    # C wrote: magic(uint32), version(int), then 10 ints
    magic = struct.unpack_from('<I', header_bytes, 0)[0]
    version = struct.unpack_from('<i', header_bytes, 4)[0]
    ints = struct.unpack_from('<10i', header_bytes, 8)
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, head_dim, shared_classifier, group_size = ints
    return Config(magic, version, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, head_dim, shared_classifier, group_size)


def _init_qtensors_from(ptr: int, n: int, size_each: int, mm: mmap.mmap) -> Tuple[List[QuantizedTensor], int]:
    res: List[QuantizedTensor] = []
    for _ in range(n):
        q = np.frombuffer(mm, dtype=np.int8, count=size_each, offset=ptr)
        ptr += size_each
        s = np.frombuffer(mm, dtype=np.float32, count=size_each // GS, offset=ptr)
        ptr += (size_each // GS) * 4
        res.append(QuantizedTensor(q=q, s=s))
    return res, ptr


def memory_map_weights(cfg: Config, mm: mmap.mmap) -> TransformerWeights:
    # fp32 blocks first
    ptr = 256
    count = cfg.n_layers * cfg.dim
    rms_att = np.frombuffer(mm, dtype=np.float32, count=count, offset=ptr)
    ptr += count * 4
    rms_ffn = np.frombuffer(mm, dtype=np.float32, count=count, offset=ptr)
    ptr += count * 4
    rms_final = np.frombuffer(mm, dtype=np.float32, count=cfg.dim, offset=ptr)
    ptr += cfg.dim * 4
    qnorm = np.frombuffer(mm, dtype=np.float32, count=cfg.n_layers * cfg.head_dim, offset=ptr)
    ptr += (cfg.n_layers * cfg.head_dim) * 4
    knorm = np.frombuffer(mm, dtype=np.float32, count=cfg.n_layers * cfg.head_dim, offset=ptr)
    ptr += (cfg.n_layers * cfg.head_dim) * 4

    # quant blocks
    # embeddings
    q_tokens_list, ptr = _init_qtensors_from(ptr, 1, cfg.vocab_size * cfg.dim, mm)
    q_tokens = q_tokens_list[0]
    # dequantize embeddings to fp32 table
    token_embedding_table = np.empty(cfg.vocab_size * cfg.dim, dtype=np.float32)
    dequantize(q_tokens, token_embedding_table)

    # attention weights
    all_heads_dim = cfg.n_heads * cfg.head_dim
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    wq, ptr = _init_qtensors_from(ptr, cfg.n_layers, cfg.dim * all_heads_dim, mm)
    wk, ptr = _init_qtensors_from(ptr, cfg.n_layers, cfg.dim * kv_dim, mm)
    wv, ptr = _init_qtensors_from(ptr, cfg.n_layers, cfg.dim * kv_dim, mm)
    wo, ptr = _init_qtensors_from(ptr, cfg.n_layers, all_heads_dim * cfg.dim, mm)

    # ffn
    w1, ptr = _init_qtensors_from(ptr, cfg.n_layers, cfg.dim * cfg.hidden_dim, mm)
    w2, ptr = _init_qtensors_from(ptr, cfg.n_layers, cfg.hidden_dim * cfg.dim, mm)
    w3, ptr = _init_qtensors_from(ptr, cfg.n_layers, cfg.dim * cfg.hidden_dim, mm)

    # classifier
    if cfg.shared_classifier:
        wcls = q_tokens
    else:
        wcls_list, ptr = _init_qtensors_from(ptr, 1, cfg.dim * cfg.vocab_size, mm)
        wcls = wcls_list[0]

    return TransformerWeights(
        rms_att_weight=rms_att,
        rms_ffn_weight=rms_ffn,
        rms_final_weight=rms_final,
        q_norm_weights=qnorm,
        k_norm_weights=knorm,
        q_tokens=q_tokens,
        token_embedding_table=token_embedding_table,
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
        w1=w1,
        w2=w2,
        w3=w3,
        wcls=wcls,
    )


def build_transformer(checkpoint_path: str, ctx_length: int = 0) -> Transformer:
    t = Transformer()
    f = open(checkpoint_path, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    header = mm[:256]
    cfg = _read_config(header)
    if cfg.magic_number != 0x616A6331:
        raise ValueError("Not a qwen3.c checkpoint")
    if cfg.version != 1:
        raise ValueError(f"Unsupported version {cfg.version}")
    if ctx_length != 0 and ctx_length <= cfg.seq_len:
        cfg.seq_len = ctx_length
    global GS
    GS = cfg.group_size
    w = memory_map_weights(cfg, mm)
    t.config = cfg
    t.weights = w
    t.state = RunState(cfg)
    t._mm = mm
    t._file = f
    return t


def free_transformer(t: Transformer):
    if t._mm is not None:
        t._mm.close()
    if t._file is not None:
        t._file.close()


# ----------------------------------------------------------------------------
# Math blocks


def rmsnorm(o: np.ndarray, x: np.ndarray, weight: np.ndarray):
    ss = float(np.dot(x, x))
    inv = 1.0 / math.sqrt((ss / x.shape[0]) + 1e-6)
    np.multiply(x, inv, out=o)
    np.multiply(o, weight, out=o)


def softmax_inplace(x: np.ndarray, size: int):
    max_val = float(np.max(x[:size]))
    x[:size] = np.exp(x[:size] - max_val)
    s = float(np.sum(x[:size]))
    x[:size] /= s


# ----------------------------------------------------------------------------
# Tokenizer


class Tokenizer:
    def __init__(self, checkpoint_path: str, vocab_size: int, enable_thinking: int):
        self.vocab_size = vocab_size
        self.vocab: List[str] = [""] * vocab_size
        self.merge_scores = np.zeros(vocab_size, dtype=np.float32)
        self.max_token_length = 0
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.prompt_template = ""
        self.system_prompt_template = ""

        tok_path = checkpoint_path + ".tokenizer"
        with open(tok_path, 'rb') as f:
            self.max_token_length = struct.unpack('<I', f.read(4))[0]
            self.bos_token_id = struct.unpack('<I', f.read(4))[0]
            self.eos_token_id = struct.unpack('<I', f.read(4))[0]
            for i in range(vocab_size):
                b = f.read(4)
                if not b:
                    self.vocab[i] = ""
                    continue
                self.merge_scores[i] = struct.unpack('<f', b)[0]
                tlen = struct.unpack('<I', f.read(4))[0]
                tok = f.read(tlen)
                self.vocab[i] = tok.decode('utf-8', errors='ignore')

        # load templates
        self.prompt_template = _load_template(checkpoint_path, False, enable_thinking)
        self.system_prompt_template = _load_template(checkpoint_path, True, enable_thinking)

    def decode(self, token: int) -> str:
        return self.vocab[token]

    def _str_lookup(self, s: str) -> int:
        for i, v in enumerate(self.vocab):
            if s == v:
                return i
        return -1

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        buf = bytearray()
        special_buf = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            found_special = False
            if ch == '<':
                end = text.find('>', i, i + 65)
                if end != -1:
                    t = text[i:end+1]
                    tid = self._str_lookup(t)
                    if tid != -1:
                        tokens.append(tid)
                        i = end + 1
                        found_special = True
                        continue
            if not found_special:
                tid = self._str_lookup(ch)
                if tid != -1:
                    tokens.append(tid)
                else:
                    # unknown char, skip but keep position for compatibility
                    tokens.append(0)
                i += 1

        # merge loop
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1
            for j in range(len(tokens) - 1):
                pair = self.vocab[tokens[j]] + self.vocab[tokens[j+1]]
                tid = self._str_lookup(pair)
                if tid != -1:
                    score = float(self.merge_scores[tid])
                    if score > best_score:
                        best_score = score
                        best_id = tid
                        best_idx = j
            if best_idx == -1:
                break
            tokens[best_idx] = best_id
            del tokens[best_idx + 1]
        return tokens


def _load_template(checkpoint_path: str, with_system: bool, enable_thinking: int) -> str:
    path = checkpoint_path
    if with_system:
        path += ".template.with-system-and-thinking" if enable_thinking else ".template.with-system"
    else:
        path += ".template.with-thinking" if enable_thinking else ".template"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ----------------------------------------------------------------------------
# Sampler


class Sampler:
    def __init__(self, vocab_size: int, temperature: float, topp: float, rng_seed: int):
        self.vocab_size = vocab_size
        self.temperature = max(0.0, float(temperature))
        self.topp = topp
        self.rng_state = np.uint64(rng_seed if rng_seed > 0 else int(time.time()))
        self._probindex = np.zeros((vocab_size, 2), dtype=np.float32)  # [prob, idx]

    def _random_u32(self) -> np.uint32:
        s = self.rng_state
        s ^= s >> np.uint64(12)
        s ^= s << np.uint64(25)
        s ^= s >> np.uint64(27)
        self.rng_state = s
        return np.uint32((s * np.uint64(0x2545F4914F6CDD1D)) >> np.uint64(32))

    def _random_f32(self) -> float:
        return float((self._random_u32() >> np.uint32(8)) / np.float32(16777216.0))

    def _sample_mult(self, p: np.ndarray) -> int:
        coin = self._random_f32()
        cdf = 0.0
        for i in range(p.shape[0]):
            cdf += float(p[i])
            if coin < cdf:
                return int(i)
        return int(p.shape[0] - 1)

    def _sample_topp(self, p: np.ndarray, topp: float) -> int:
        cutoff = (1.0 - topp) / max(1, p.shape[0] - 1)
        idx = np.where(p >= cutoff)[0]
        if idx.size == 0:
            idx = np.arange(p.shape[0])
        probs = p[idx]
        order = np.argsort(-probs)
        idx = idx[order]
        probs = probs[order]
        cumulative = np.cumsum(probs)
        last_idx = np.searchsorted(cumulative, topp, side='right')
        last_idx = min(last_idx, probs.shape[0] - 1)
        cumulative_prob = cumulative[last_idx]
        r = self._random_f32() * float(cumulative_prob)
        cdf = 0.0
        for i in range(last_idx + 1):
            cdf += float(probs[i])
            if r < cdf:
                return int(idx[i])
        return int(idx[last_idx])

    def sample(self, logits: np.ndarray) -> int:
        if self.temperature == 0.0:
            return int(np.argmax(logits))
        scaled = logits / self.temperature
        # softmax
        m = float(np.max(scaled))
        probs = np.exp(scaled - m)
        probs /= float(np.sum(probs))
        if self.topp <= 0.0 or self.topp >= 1.0:
            return self._sample_mult(probs)
        else:
            return self._sample_topp(probs, self.topp)


# ----------------------------------------------------------------------------
# Forward pass


def forward(transformer: Transformer, token: int, pos: int) -> np.ndarray:
    cfg = transformer.config
    w = transformer.weights
    s = transformer.state
    assert cfg and w and s
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    kv_mul = cfg.n_heads // cfg.n_kv_heads
    all_heads_dim = cfg.n_heads * cfg.head_dim

    # x = embedding[token]
    s.x[:] = w.token_embedding_table[token * cfg.dim : (token + 1) * cfg.dim]

    for l in range(cfg.n_layers):
        loff = l * cfg.seq_len * kv_dim
        k_ptr = s.key_cache[loff + pos * kv_dim : loff + (pos + 1) * kv_dim]
        v_ptr = s.value_cache[loff + pos * kv_dim : loff + (pos + 1) * kv_dim]

        # attn rmsnorm (write only first dim elements)
        rmsnorm(s.xb[:cfg.dim], s.x, w.rms_att_weight[l * cfg.dim : (l + 1) * cfg.dim])

        # qkv
        quantize(s.xq, s.xb[:cfg.dim])
        q8_matmul(s.q, s.xq, w.wq[l], cfg.dim, all_heads_dim)
        q8_matmul(k_ptr, s.xq, w.wk[l], cfg.dim, kv_dim)
        q8_matmul(v_ptr, s.xq, w.wv[l], cfg.dim, kv_dim)

        # Q- & K- RMSNorm + RoPE
        # queries
        for h in range(cfg.n_heads):
            q = s.q[h * cfg.head_dim : (h + 1) * cfg.head_dim]
            q_w = w.q_norm_weights[l * cfg.head_dim : (l + 1) * cfg.head_dim]
            rmsnorm(q, q, q_w)
            half = cfg.head_dim // 2
            for j in range(half):
                freq = pow(1e6, -float(j) / float(half))
                cos_f = math.cos(pos * freq)
                sin_f = math.sin(pos * freq)
                xr = q[j]
                xi = q[j + half]
                q[j] = xr * cos_f - xi * sin_f
                q[j + half] = xr * sin_f + xi * cos_f

        # keys (in-place in cache slice)
        for h in range(cfg.n_kv_heads):
            k = k_ptr[h * cfg.head_dim : (h + 1) * cfg.head_dim]
            k_w = w.k_norm_weights[l * cfg.head_dim : (l + 1) * cfg.head_dim]
            rmsnorm(k, k, k_w)
            half = cfg.head_dim // 2
            for j in range(half):
                freq = pow(1e6, -float(j) / float(half))
                cos_f = math.cos(pos * freq)
                sin_f = math.sin(pos * freq)
                xr = k[j]
                xi = k[j + half]
                k[j] = xr * cos_f - xi * sin_f
                k[j + half] = xr * sin_f + xi * cos_f

        # attention heads
        for h in range(cfg.n_heads):
            q = s.q[h * cfg.head_dim : (h + 1) * cfg.head_dim]
            att = s.att[h * cfg.seq_len : (h + 1) * cfg.seq_len]
            for t in range(pos + 1):
                k = s.key_cache[loff + t * kv_dim + (h // kv_mul) * cfg.head_dim : loff + t * kv_dim + (h // kv_mul + 1) * cfg.head_dim]
                score = float(np.dot(q, k)) / math.sqrt(cfg.head_dim)
                att[t] = score
            softmax_inplace(att, pos + 1)
            xb = s.xb[h * cfg.head_dim : (h + 1) * cfg.head_dim]
            xb.fill(0.0)
            for t in range(pos + 1):
                v = s.value_cache[loff + t * kv_dim + (h // kv_mul) * cfg.head_dim : loff + t * kv_dim + (h // kv_mul + 1) * cfg.head_dim]
                xb += att[t] * v

        # project out
        quantize(s.xq, s.xb)
        q8_matmul(s.xb, s.xq, w.wo[l], all_heads_dim, cfg.dim)
        s.x += s.xb

        # ffn (norm on first dim)
        rmsnorm(s.xb[:cfg.dim], s.x, w.rms_ffn_weight[l * cfg.dim : (l + 1) * cfg.dim])
        quantize(s.xq, s.xb[:cfg.dim])
        q8_matmul(s.hb, s.xq, w.w1[l], cfg.dim, cfg.hidden_dim)
        q8_matmul(s.hb2, s.xq, w.w3[l], cfg.dim, cfg.hidden_dim)
        # SwiGLU: hb = silu(hb) * hb2
        # silu(x)=x*σ(x)
        s.hb = s.hb * s.hb2 * (1.0 / (1.0 + np.exp(-s.hb)))
        quantize(s.hq, s.hb)
        q8_matmul(s.xb, s.hq, w.w2[l], cfg.hidden_dim, cfg.dim)
        s.x += s.xb

    # final norm
    rmsnorm(s.x, s.x, w.rms_final_weight)
    quantize(s.xq, s.x)
    q8_matmul(s.logits, s.xq, w.wcls, cfg.dim, cfg.vocab_size)
    return s.logits


# ----------------------------------------------------------------------------
# Loops


def generate(transformer: Transformer, tokenizer: Tokenizer, sampler: Sampler, prompt: Optional[str]):
    if prompt is None:
        prompt = ""
    prompt_tokens = tokenizer.encode(prompt)
    if len(prompt_tokens) < 1:
        print("Please provide a prompt via -i <string>.")
        sys.exit(1)
    token = prompt_tokens[0]
    pos = 0
    next_tok = 0
    while pos < transformer.config.seq_len:
        logits = forward(transformer, token, pos)
        if pos < len(prompt_tokens) - 1:
            next_tok = prompt_tokens[pos + 1]
        else:
            next_tok = sampler.sample(logits)
        print(tokenizer.decode(token), end="", flush=True)
        token = next_tok
        pos += 1
        if pos >= len(prompt_tokens) and (next_tok == tokenizer.bos_token_id or next_tok == tokenizer.eos_token_id):
            break
    print()


def chat(transformer: Transformer, tokenizer: Tokenizer, sampler: Sampler, cli_user_prompt: Optional[str], system_prompt: Optional[str]):
    PROMPT_BUFFER_SIZE = 32768
    user_turn = True
    pos = 0
    next_tok = 0
    prompt_tokens: List[int] = []
    while True:
        if pos >= transformer.config.seq_len:
            print("\n(context window full, clearing)")
            user_turn = True
            pos = 0
        if user_turn:
            if cli_user_prompt is not None:
                if pos > 0:
                    break
                user_prompt = cli_user_prompt
            else:
                try:
                    user_prompt = input("\n> ")
                except EOFError:
                    break
                if len(user_prompt) == 0:
                    break
            if pos == 0 and system_prompt:
                rendered = tokenizer.system_prompt_template % (system_prompt, user_prompt)
            else:
                rendered = tokenizer.prompt_template % (user_prompt)
            prompt_tokens = tokenizer.encode(rendered)
            pos = 0
            user_turn = False
        token = prompt_tokens[pos] if pos < len(prompt_tokens) else next_tok
        logits = forward(transformer, token, pos)
        pos += 1
        next_tok = sampler.sample(logits)
        if pos >= len(prompt_tokens):
            if token == tokenizer.bos_token_id or token == tokenizer.eos_token_id:
                print()
                user_turn = True
            elif next_tok != tokenizer.bos_token_id and next_tok != tokenizer.eos_token_id:
                print(tokenizer.decode(next_tok), end="", flush=True)


# ----------------------------------------------------------------------------
# CLI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='path to .bin checkpoint')
    parser.add_argument('-t', type=float, default=1.0, help='temperature')
    parser.add_argument('-p', type=float, default=0.9, help='top-p')
    parser.add_argument('-s', type=int, default=0, help='random seed (0=time)')
    parser.add_argument('-c', type=int, default=0, help='context length (0=max)')
    parser.add_argument('-m', type=str, default='chat', help='mode: generate|chat')
    parser.add_argument('-i', type=str, default=None, help='input prompt')
    parser.add_argument('-y', type=str, default=None, help='system prompt (chat)')
    parser.add_argument('-r', type=int, default=0, help='reasoning mode (0|1)')
    args = parser.parse_args()

    t = build_transformer(args.checkpoint, args.c)
    tokenizer = Tokenizer(args.checkpoint, t.config.vocab_size, args.r)
    sampler = Sampler(t.config.vocab_size, args.t, args.p, args.s)

    if args.i is None:
        print(f"hidden_size={t.config.dim}, intermediate_size={t.config.hidden_dim}, num_hidden_layers={t.config.n_layers}, num_attention_heads={t.config.n_heads}, num_kv_heads={t.config.n_kv_heads}, head_dim={t.config.head_dim}, ctx_length={t.config.seq_len}, vocab_size={t.config.vocab_size}, shared_classifier={t.config.shared_classifier}, quantization_block_size={t.config.group_size}")

    if args.m == 'generate':
        generate(t, tokenizer, sampler, args.i)
    elif args.m == 'chat':
        chat(t, tokenizer, sampler, args.i, args.y)
    else:
        print(f"Unknown mode: {args.m}")
        sys.exit(1)

    free_transformer(t)


if __name__ == '__main__':
    main()

