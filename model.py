from dataclasses import dataclass
import torch
from torch.nn import functional as F
import torch.nn as nn
import math

# ============================================

@dataclass
class GPTConfig:
    block_size: int = 1024   # Max sequence length
    vocab_size: int = 50257  # Number of tokens: 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_embd: int = 768        # Embedding dimension
    n_layer: int = 12        # Number of layers
    n_head: int = 12         # Number of attention heads



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # qkv 3 components
        # output projection
        self.c_proj = nn.Linear(config.n_embd , config.n_embd)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # not really 'bias' more of the trick trick to consistent
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        # Shape x: [B, T, C] 
        B,T,C = x.shape
        qkv = self.c_attn(x)

        q,k,v = torch.split(qkv, self.n_embd, dim = 2)
        # Shape q: [B, T, 3*C]  -> [B, T, C] [B, T, C] [B, T, C]

        # attention matrix compute for different heads
        q = q.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
        #  [B,T,C]  -> [B,T, n_head, C//n_head] -> [B, n_head, T, C//n_head]
        k = k.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C // self.n_head).transpose(1,2)

        # attention is [T,T] matrix for all query key matrix
        att = (q @ k.transpose(-2, -1) ) * (1.0 / math.sqrt(self.n_embd))
         # [B, n_head, T, C//n_head] * [B, n_head, C//n_head, T] = [B, n_head, T, T]

        # masking to avoid looking at future tokens
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  # [B, n_head, T, T]
        
        assert  torch.allclose(att[0][0].sum(),torch.tensor(1.0*att.shape[-1]))
        # Sum of all column and then rows to number 
        # attention weighted sum
        y = att @ v  # [B, n_head, T, T] * [B, n_head, T, C//n_head] = [B, n_head, T, C//n_head]
        y = y.transpose(1,2).contiguous().view(B,T,C)
        # [B, n_head, T, C//n_head] -> [B, T, n_head, C//n_head] -> [B, T, C]
        # re-assemble all head output side-by-side -> [B, T, C]
        
        assert y.shape == x.shape
        
        return self.c_proj(y)





class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print(f"Loading pretrained model {model_type} from HuggingFace")
        
        # Design the model config, based on model_type
        config_args = {
            "gpt2":        dict(n_layer=12, n_embd=768, n_head=12),  # 124 M params
            "gpt2-medium": dict(n_layer=24, n_embd=1024, n_head=16), # 350 M params
            "gpt2-large":  dict(n_layer=36, n_embd=1280, n_head=20), # 774 M params
            "gpt2-xl":     dict(n_layer=48, n_embd=1600, n_head=25), # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard this mark/buffer parameters
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        
        sd_hf = GPT2LMHeadModel.from_pretrained(model_type).state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

def test_causal_attention():
    config = GPTConfig()
    model = CausalSelfAttention(config)
    x = torch.randn(4, 10, 768)
    y = model(x)
    print(f"Input shape: {x.shape}, output shape: {y.shape}")
    assert y.shape == x.shape

def test_block():
    config = GPTConfig()
    model = Block(config)
    x = torch.randn(4, 10, 768)
    y = model(x)
    print(f"Input shape: {x.shape}, output shape: {y.shape}")
    assert y.shape == x.shape

def test_model():
    config = GPTConfig()
    model = GPT(config)
    B, T = 4, 10
    x = torch.randint(0, 50257, (B, T))
    y, loss = model(x)
    assert y.shape == (B, T, 50257)
    print(f"Input shape: {x.shape}, output shape: {y.shape}")
    

def test_transformer():
    config = GPTConfig()
    model = GPT(config)
    B, T = 4, 10
    x = torch.randint(0, 50257, (B, T))
    y, loss = model(x)
    assert y.shape == (B, T, 50257)
    print(f"Input shape: {x.shape}, output shape: {y.shape}")

if __name__ == "__main__":
    test_causal_attention()
    test_block()
    test_transformer()
    
    model = GPT.from_pretrained("gpt2")
    print("All tests pass")
    
    test_model()