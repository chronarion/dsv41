# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import array
import collections
import copy
import re
import threading
from typing import Any

from transformers import BatchEncoding, TokenizersBackend

from vllm.renderers.chat_utils import ChatCompletionMessageParam

from .deepseek_v41_encoding import (
    IMAGE_PLACEHOLDER,
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
)
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike


def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    result = [dict(message) for message in copy.deepcopy(messages)]
    for message in result:
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {role}")
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    parts.append(block.get("text", ""))
                elif part_type in ("image_url", "input_image", "image_pil"):
                    parts.append(IMAGE_PLACEHOLDER)
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            message["content"] = "\n\n".join(parts)
    return result


# dsv41 A3: the chat path tokenizes the rendered prompt one added-token-delimited segment at a time, with an LRU cache
# of segments, and encodes a long uncached segment as parallel chunks (encode_batch). The tokenizer has an empty
# normalizer and no added token strips whitespace, so added tokens are extracted (leftmost-longest) before
# pre-tokenizing and the text between them encodes independently: the ids equal encode(prompt). A chunk boundary sits
# only where no pre-tokenizer pattern can span it (a space between two ASCII letters, or a newline followed by an
# ASCII letter), so the chunks' ids concatenate to the segment's ids too. Agent conversations grow append-only and
# clients resend the same prompt, so a request re-tokenizes only its new segments; a long new segment is ~5x faster.
_CUT = re.compile(r"(?<=[A-Za-z]) (?=[A-Za-z])|(?<=\n)(?=[A-Za-z])")
_CHUNK_CHARS = 32_000


class _SegmentEncoder:
    def __init__(self, tokenizer, added_vocab: dict[str, int], max_cached_tokens: int = 8_000_000) -> None:
        # its own Rust tokenizer: HF's __call__ leaves truncation / padding set on the shared backend
        # (enable_truncation(max_length) persists between calls), which would truncate these encodes silently
        from tokenizers import Tokenizer

        self._backend = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
        self._backend.no_truncation()
        self._backend.no_padding()
        self._added = dict(added_vocab)
        self._split = re.compile("(" + "|".join(map(re.escape, sorted(self._added, key=len, reverse=True))) + ")")
        self._cache: collections.OrderedDict[str, array.array] = collections.OrderedDict()
        self._cached_tokens = 0
        self._max_cached_tokens = max_cached_tokens
        self._lock = threading.Lock()

    def encode(self, prompt: str) -> list[int]:
        ids: list[int] = []
        for index, part in enumerate(self._split.split(prompt)):
            if index % 2:
                ids.append(self._added[part])
            elif part:
                ids.extend(self._segment(part))
        return ids

    def _chunks(self, text: str) -> list[str]:
        n = len(text) // _CHUNK_CHARS
        if n < 2:
            return [text]
        step, out, start = len(text) // n, [], 0
        for k in range(1, n):
            m = _CUT.search(text, max(start + 1, k * step))
            if m is None:
                break
            out.append(text[start:m.start()])
            start = m.start()
        out.append(text[start:])
        return out

    def _segment(self, text: str) -> array.array:
        with self._lock:
            hit = self._cache.get(text)
            if hit is not None:
                self._cache.move_to_end(text)
                return hit
        chunks = self._chunks(text)
        if len(chunks) == 1:
            ids = array.array("i", self._backend.encode(text, add_special_tokens=False).ids)
        else:
            ids = array.array("i")
            for e in self._backend.encode_batch(chunks, add_special_tokens=False):
                ids.extend(e.ids)
        with self._lock:
            if text not in self._cache:
                self._cache[text] = ids
                self._cached_tokens += len(ids)
                while self._cached_tokens > self._max_cached_tokens and self._cache:
                    _, evicted = self._cache.popitem(last=False)
                    self._cached_tokens -= len(evicted)
        return ids


def get_deepseek_v41_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """Wrap an HF tokenizer with the V4.1 prompt encoder."""
    wrapped = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()
    segments = _SegmentEncoder(tokenizer, added_vocab)  # dsv41 A3

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list[ChatCompletionMessageParam],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> str | list[int]:
            # The generic renderer's conversation has already flattened text
            # parts with '\n'. V4.1 encodes the original messages with '\n\n'.
            conversation = _normalize_messages(messages)
            if tools:
                system = next((m for m in conversation if m["role"] == "system"), None)
                if system is None:
                    system = {"role": "system", "content": ""}
                    conversation.insert(0, system)
                system["tools"] = tools

            thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
            if "thinking" not in kwargs and "enable_thinking" not in kwargs:
                thinking = True
            effort = kwargs.get("reasoning_effort")
            if effort == "none":
                thinking = False
                effort = None
            if effort is None:
                effort = "high"
            if not (
                (type(effort) is int and 1 <= effort <= 100)
                or (isinstance(effort, str) and effort in REASONING_EFFORT_MAPPINGS)
            ):
                raise ValueError(
                    "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, "
                    "or an integer within [1, 100] in chat_template_kwargs"
                )

            prompt = encode_messages(
                conversation,
                thinking_mode="thinking" if thinking else "chat",
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=effort,
            )
            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {
                    key: kwargs[key]
                    for key in ("truncation", "max_length")
                    if key in kwargs
                }
                if tokenizer_kwargs:
                    return self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)
                return segments.encode(prompt)  # dsv41 A3: same ids as self.encode(prompt, add_special_tokens=False)
            return prompt

        def __call__(self, text=None, *args, **kwargs):
            # dsv41 A3: the renderer tokenizes the rendered chat prompt here (tokenizer(prompt, truncation=...,
            # max_length=..., add_special_tokens=False)); the segment encoder returns the same input_ids
            if (isinstance(text, str) and not args and kwargs.get("add_special_tokens") is False
                    and set(kwargs) <= {"add_special_tokens", "truncation", "max_length"}):
                ids = segments.encode(text)
                limit = kwargs.get("max_length")
                if not (kwargs.get("truncation") and limit is not None and len(ids) > limit):
                    return BatchEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)})
            return super().__call__(text, *args, **kwargs)

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v41_tokenizer, (tokenizer,)

    _DeepseekV41Tokenizer.__name__ = f"DSV41{tokenizer.__class__.__name__}"
    wrapped.__class__ = _DeepseekV41Tokenizer
    return wrapped


class DeepseekV41Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v41_tokenizer(tokenizer))
