from typing import List, Optional
from emt.cache import RotatingBufferCache
from emt.utils import sample
import torch
from pathlib import Path
from modal import Secret, enter, method

from image import GPU_CONFIG, MODEL_DIR, stub

from emt.model import Transformer
from emt.tokenizer import Tokenizer


@stub.cls(
    gpu=GPU_CONFIG, 
    secrets=[Secret.from_name("huggingface-secret")]
)
class Mistral7B:
    @enter()
    def load_model(self):

        # Load the model. Tip: MPT models may require `trust_remote_code=true`.
        self.tokenizer = Tokenizer(str(Path(MODEL_DIR) / "tokenizer.model"))
        self.transformer = Transformer.from_folder(Path(MODEL_DIR), max_batch_size=3, device="cuda", dtype=torch.float16)
        
        self.template = """<s>[INST] <<SYS>>{system}<</SYS>>{user} [/INST] """

    @method()
    def generate(self, user_questions: str):
        import time

        # prompts = [
        #     self.template.format(system="", user=q) for q in user_questions
        # ]
        
        start = time.monotonic_ns()
        res, _logprobs = self.inference(
            [
                "This is a test",
                "This is another test",
                "This is a third test, mistral AI is very good at testing. ",
            ],
            self.transformer,
            self.tokenizer,
            max_tokens=256,
            temperature=0.7,
        )
        for x in res:
            print(x)
            print("=====================")
        end = time.monotonic_ns()
        print(f"Time taken: {end - start} ns")
        
    @torch.inference_mode()
    def inference(
        self,
        prompts: List[str],
        model: Transformer,
        tokenizer: Tokenizer,
        *,
        max_tokens: int,
        chunk_size: Optional[int] = None,
        temperature: float = 0.7
    ):
        model = model.eval()
        B, V = len(prompts), model.args.vocab_size

        # Tokenize
        encoded_prompts = [tokenizer.encode(prompt, bos=True) for prompt in prompts]
        seqlens = [len(x) for x in encoded_prompts]

        # Cache
        cache_window = min(model.args.sliding_window, max(seqlens) + max_tokens)
        cache = RotatingBufferCache(
            model.args.n_layers,
            model.args.max_batch_size,
            cache_window,
            model.args.n_kv_heads,
            model.args.head_dim,
        )
        cache.to(device=model.device, dtype=model.dtype)
        cache.reset()

        # Bookkeeping
        logprobs: list[list[float]] = [[] for _ in range(B)]
        last_token_prelogits = None

        # One chunk if size not specified
        max_prompt_len = max(seqlens)
        if chunk_size is None:
            chunk_size = max_prompt_len

        # Encode prompt by chunks
        for s in range(0, max_prompt_len, chunk_size):
            prompt_chunks = [p[s : s + chunk_size] for p in encoded_prompts]
            assert all(len(p) > 0 for p in prompt_chunks)
            prelogits = model.forward(
                torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
                cache=cache,
                seqlens=[len(p) for p in prompt_chunks],
            )
            logits = torch.log_softmax(prelogits, dim=-1)

            if last_token_prelogits is not None:
                # Pass > 1
                last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
                for i_seq in range(B):
                    logprobs[i_seq].append(
                        last_token_logits[i_seq, prompt_chunks[i_seq][0]].item()
                    )

            offset = 0
            for i_seq, sequence in enumerate(prompt_chunks):
                logprobs[i_seq].extend(
                    [
                        logits[offset + i, sequence[i + 1]].item()
                        for i in range(len(sequence) - 1)
                    ]
                )
                offset += len(sequence)

            last_token_prelogits = prelogits.index_select(
                0,
                torch.tensor(
                    [len(p) for p in prompt_chunks], device=prelogits.device
                ).cumsum(dim=0)
                - 1,
            )
            assert last_token_prelogits.shape == (B, V)

        # decode
        generated_tokens_list = []
        for i_token in range(max_tokens):
            assert last_token_prelogits is not None
            next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)

            last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
            for i in range(B):
                logprobs[i].append(last_token_logits[i, next_token[i]].item())

            generated_tokens_list.append(next_token[:, None])
            last_token_prelogits = model.forward(
                next_token, cache=cache, seqlens=[1] * len(prompts)
            )
            assert last_token_prelogits.shape == (B, V)

        generated_words = []
        if generated_tokens_list:
            generated_tokens = torch.cat(generated_tokens_list, 1)
            for i, x in enumerate(encoded_prompts):
                generated_words.append(tokenizer.decode(x + generated_tokens[i].tolist()))

        return generated_words, logprobs

