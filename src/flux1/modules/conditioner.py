from pathlib import Path

from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):
    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        is_clip: bool,
        max_length: int,
        **hf_kwargs,
    ):
        super().__init__()
        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path)
        if not model_path.is_dir() or not tokenizer_path.is_dir():
            raise FileNotFoundError(
                f"Missing local text encoder or tokenizer: {model_path}, {tokenizer_path}"
            )
        self.is_clip = is_clip
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                str(tokenizer_path), max_length=max_length, local_files_only=True
            )
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                str(model_path), local_files_only=True, **hf_kwargs
            )
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                str(tokenizer_path), max_length=max_length, local_files_only=True
            )
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(
                str(model_path), local_files_only=True, **hf_kwargs
            )

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key].bfloat16()
