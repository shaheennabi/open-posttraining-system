
# Qwen3 Tokenizer: Byte-Level BPE Implementation

import unicodedata
import regex
import re
import json
from pathlib import Path

try:
    from jinja2 import Environment
except ImportError:
    Environment = None


TOKENIZER_PATH = Path(__file__).resolve().parent / "qwen" / "tokenizer.json"
TOKENIZER_CONFIG_PATH = TOKENIZER_PATH.with_name("tokenizer_config.json")


# NFC unicode normalization
def normalize_text(text):
    return unicodedata.normalize("NFC", text)


# Qwen3 pre-tokenization
PRETOKEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    r"[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|"
    r"\s+(?!\S)|"
    r"\s+"
)

PRETOKEN_RE = regex.compile(PRETOKEN_PATTERN)


def pre_tokenize(text):
    return [match.group() for match in PRETOKEN_RE.finditer(text)]


# Byte-level encoding
def bytes_to_unicode():

    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )

    cs = bs[:]
    n = 0

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return dict(zip(bs, map(chr, cs)))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


def byte_level_encode(text):
    return "".join(BYTE_ENCODER[b] for b in text.encode("utf-8"))


# Load tokenizer JSON files
def loading_tokenizer(path):

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def loading_tokenizer_config(path):

    path = Path(path)

    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


class Qwen3Tokenizer:

    def __init__(
        self,
        tokenizer_path=TOKENIZER_PATH,
        apply_chat_template=False,
        add_generation_prompt=False,
        add_thinking=False,
        tokenizer_config_path=None
    ):

        # Load tokenizer data and configuration
        self.tokenizer_data = loading_tokenizer(tokenizer_path)

        if tokenizer_config_path is None:
            tokenizer_config_path = Path(tokenizer_path).with_name(
                "tokenizer_config.json"
            )

        self.tokenizer_config = loading_tokenizer_config(
            tokenizer_config_path
        )

        # Chat defaults (compatible with previous interface)
        self.default_chat_wrapping = apply_chat_template
        self.default_add_generation_prompt = add_generation_prompt
        self.default_enable_thinking = add_thinking

        # Vocabulary and reverse vocabulary
        self.vocabulary = self.tokenizer_data["model"]["vocab"]

        self.reverse_vocabulary = {
            token_id: token
            for token, token_id in self.vocabulary.items()
        }

        # Merge ranks
        self.merge_ranks = {}

        for rank, merge in enumerate(self.tokenizer_data["model"]["merges"]):

            if isinstance(merge, str):
                merge = merge.split(" ", 1)

            self.merge_ranks[tuple(merge)] = rank

        # Added tokens
        added_tokens = self.tokenizer_data.get("added_tokens", [])

        # Fall back to tokenizer_config if necessary
        if not added_tokens:

            added_tokens = []

            for token_id, token in self.tokenizer_config.get(
                "added_tokens_decoder", {}
            ).items():

                entry = dict(token)
                entry["id"] = int(token_id)
                added_tokens.append(entry)

        self.added_token_to_id = {
            token["content"]: token["id"]
            for token in added_tokens
        }

        self.id_to_added_token = {
            token["id"]: token["content"]
            for token in added_tokens
        }

        # Special tokens
        special_token_strings = {
            token["content"]
            for token in added_tokens
            if token.get("special", False)
        }

        self.additional_special_tokens = [
            token["content"] if isinstance(token, dict) else token
            for token in self.tokenizer_config.get(
                "additional_special_tokens", []
            )
        ]

        special_token_strings.update(self.additional_special_tokens)

        self.special_tokens = {
            token: self._get_token_id(token)
            for token in special_token_strings
            if self._get_token_id(token) is not None
        }

        self.special_token_ids = set(self.special_tokens.values())

        # Configuration
        self.add_bos_token = self.tokenizer_config.get("add_bos_token", False)
        self.add_eos_token = self.tokenizer_config.get("add_eos_token", False)
        self.add_prefix_space = self.tokenizer_config.get("add_prefix_space", False)

        self.bos_token = self._get_token_content(
            self.tokenizer_config.get("bos_token")
        )

        self.eos_token = self._get_token_content(
            self.tokenizer_config.get("eos_token")
        )

        self.pad_token = self._get_token_content(
            self.tokenizer_config.get("pad_token")
        )

        self.unk_token = self._get_token_content(
            self.tokenizer_config.get("unk_token")
        )

        # Token IDs
        self.bos_token_id = self._get_token_id(self.bos_token)
        self.eos_token_id = self._get_token_id(self.eos_token)
        self.pad_token_id = self._get_token_id(self.pad_token)
        self.unk_token_id = self._get_token_id(self.unk_token)

        self.endoftext_token_id = self._get_token_id("<|endoftext|>")

        # Generation stopping tokens
        self.stop_token_ids = {
            token_id
            for token_id in (
                self.eos_token_id,
                self.endoftext_token_id
            )
            if token_id is not None
        }

        self.model_max_length = self.tokenizer_config.get(
            "model_max_length"
        )

        self.padding_side = self.tokenizer_config.get(
            "padding_side", "right"
        )

        self.truncation_side = self.tokenizer_config.get(
            "truncation_side", "right"
        )

        self.errors = self.tokenizer_config.get("errors", "replace")

        self.clean_up_tokenization_spaces = self.tokenizer_config.get(
            "clean_up_tokenization_spaces", False
        )

        self.tokenizer_class = self.tokenizer_config.get(
            "tokenizer_class"
        )

        # Chat template
        self.chat_template = self.tokenizer_config.get("chat_template")

        self.chat_template_renderer = None

        if self.chat_template and Environment is not None:
            self.chat_template_renderer = Environment(
                autoescape=False
            ).from_string(self.chat_template)

        # Split around all added tokens, longest first
        added_token_strings = sorted(
            self.added_token_to_id,
            key=len,
            reverse=True
        )

        if added_token_strings:

            pattern = "(" + "|".join(
                re.escape(token) for token in added_token_strings
            ) + ")"

            self.added_token_pattern = re.compile(pattern)

        else:
            self.added_token_pattern = None


    # Get token content from configuration
    @staticmethod
    def _get_token_content(token):

        if isinstance(token, dict):
            return token.get("content")

        return token


    # Get token ID
    def _get_token_id(self, token):

        if token is None:
            return None

        return self.added_token_to_id.get(
            token,
            self.vocabulary.get(token)
        )


    # BPE merge algorithm
    def apply_bpe(self, byte_piece):

        symbols = list(byte_piece)

        while len(symbols) > 1:

            # Find valid adjacent pairs
            available_pairs = {
                (symbols[i], symbols[i + 1])
                for i in range(len(symbols) - 1)
                if (symbols[i], symbols[i + 1]) in self.merge_ranks
            }

            if not available_pairs:
                break

            # Merge the pair with the lowest rank
            best_pair = min(
                available_pairs,
                key=self.merge_ranks.__getitem__
            )

            merged_symbols = []
            i = 0

            # Merge all non-overlapping occurrences
            while i < len(symbols):

                if (
                    i + 1 < len(symbols)
                    and (symbols[i], symbols[i + 1]) == best_pair
                ):
                    merged_symbols.append(symbols[i] + symbols[i + 1])
                    i += 2

                else:
                    merged_symbols.append(symbols[i])
                    i += 1

            symbols = merged_symbols

        return symbols


    # Encode text into token IDs
    def encode(self, text, chat_wrapped=None):

        if not isinstance(text, str):
            raise TypeError("encode() expects a string")

        # Use constructor default when chat_wrapped is not specified
        if chat_wrapped is None:
            chat_wrapped = self.default_chat_wrapping

        if chat_wrapped:
            text = self._wrap_chat(
                text,
                add_generation_prompt=self.default_add_generation_prompt,
                enable_thinking=self.default_enable_thinking
            )

        if self.add_prefix_space and text and not text.startswith(" "):
            text = " " + text

        text = normalize_text(text)

        # Separate added tokens from ordinary text
        if self.added_token_pattern:
            parts = [
                part for part in self.added_token_pattern.split(text)
                if part
            ]
        else:
            parts = [text]

        token_ids = []

        for part in parts:

            # Added tokens map directly to IDs
            if part in self.added_token_to_id:
                token_ids.append(self.added_token_to_id[part])

            else:

                # Ordinary text -> pre-tokenization -> bytes -> BPE
                for piece in pre_tokenize(part):

                    byte_piece = byte_level_encode(piece)

                    bpe_tokens = self.apply_bpe(byte_piece)

                    for token in bpe_tokens:
                        if token not in self.vocabulary:
                            raise KeyError(
                                f"BPE token not found in vocabulary: {token!r}"
                            )

                        token_ids.append(self.vocabulary[token])

        # Add configured BOS/EOS tokens when enabled
        if self.add_bos_token and self.bos_token_id is not None:
            token_ids.insert(0, self.bos_token_id)

        if self.add_eos_token and self.eos_token_id is not None:
            token_ids.append(self.eos_token_id)

        return token_ids


    # Decode token IDs back into text
    def decode(self, token_ids, skip_special_tokens=False):

        decoded_text = []
        byte_buffer = []

        for token_id in token_ids:

            token_id = int(token_id)

            # Added tokens are decoded directly
            if token_id in self.id_to_added_token:

                # Decode preceding ordinary tokens first
                if byte_buffer:
                    decoded_text.append(
                        bytes(byte_buffer).decode(
                            "utf-8",
                            errors=self.errors
                        )
                    )
                    byte_buffer = []

                # Optionally skip special tokens
                if not (
                    skip_special_tokens
                    and token_id in self.special_token_ids
                ):
                    decoded_text.append(
                        self.id_to_added_token[token_id]
                    )

            else:

                # Convert ordinary BPE tokens back into bytes
                if token_id not in self.reverse_vocabulary:
                    raise ValueError(f"Unknown token ID: {token_id}")

                token = self.reverse_vocabulary[token_id]

                byte_buffer.extend(
                    BYTE_DECODER[char] for char in token
                )

        # Decode remaining bytes
        if byte_buffer:
            decoded_text.append(
                bytes(byte_buffer).decode(
                    "utf-8",
                    errors=self.errors
                )
            )

        return "".join(decoded_text)


    # Apply Qwen3 chat template
    def apply_chat_template(
        self,
        messages,
        tools=None,
        add_generation_prompt=False,
        enable_thinking=None
    ):

        if not self.chat_template:
            raise ValueError("No chat template found in tokenizer configuration")

        if self.chat_template_renderer is None:
            raise ImportError(
                "Jinja2 is required for chat formatting. Install it using: pip install jinja2"
            )

        # Support a single user message
        if isinstance(messages, str):
            messages = [
                {"role": "user", "content": messages}
            ]

        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("messages must be a non-empty list or string")

        # Render the configured Qwen3 template
        template_inputs = {
            "messages": messages,
            "tools": tools or [],
            "add_generation_prompt": add_generation_prompt
        }

        if enable_thinking is not None:
            template_inputs["enable_thinking"] = enable_thinking

        return self.chat_template_renderer.render(**template_inputs)


    # Wrap a single user message (previous interface)
    def _wrap_chat(
        self,
        user_msg,
        add_generation_prompt=None,
        enable_thinking=None,
        tools=None
    ):

        if add_generation_prompt is None:
            add_generation_prompt = self.default_add_generation_prompt

        if enable_thinking is None:
            enable_thinking = self.default_enable_thinking

        return self.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking
        )


    # Format a conversation and encode it
    def encode_chat(
        self,
        user_msg,
        add_generation_prompt=True,
        enable_thinking=False,
        tools=None
    ):

        # Support both a string and a list of messages
        if isinstance(user_msg, str):

            formatted_text = self._wrap_chat(
                user_msg,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                tools=tools
            )

        else:

            formatted_text = self.apply_chat_template(
                user_msg,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking
            )

        return self.encode(formatted_text, chat_wrapped=False)


    # Batch encoding, padding and attention masks
    def encode_batch(
        self,
        texts,
        padding=True,
        truncation=False,
        max_length=None,
        return_attention_mask=True,
        padding_side=None,
        truncation_side=None,
        pad_to_multiple_of=None
    ):

        if isinstance(texts, str) or not isinstance(texts, (list, tuple)):
            raise TypeError("encode_batch() expects a list or tuple of strings")

        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Every item in texts must be a string")

        # Validate options
        if padding not in (True, False, "longest", "max_length", None):
            raise ValueError(
                "padding must be True, False, 'longest', or 'max_length'"
            )

        if not isinstance(truncation, bool):
            raise TypeError("truncation must be a boolean")

        padding_side = padding_side or self.padding_side
        truncation_side = truncation_side or self.truncation_side

        if padding_side not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'")

        if truncation_side not in ("left", "right"):
            raise ValueError("truncation_side must be 'left' or 'right'")

        if not isinstance(return_attention_mask, bool):
            raise TypeError("return_attention_mask must be a boolean")

        if pad_to_multiple_of is not None:
            if (
                not isinstance(pad_to_multiple_of, int)
                or isinstance(pad_to_multiple_of, bool)
                or pad_to_multiple_of <= 0
            ):
                raise ValueError("pad_to_multiple_of must be a positive integer")

        # Use model maximum length when truncation is enabled
        if truncation and max_length is None:
            max_length = self.model_max_length

        if truncation and max_length is None:
            raise ValueError("max_length is required for truncation")

        if max_length is not None:
            if (
                not isinstance(max_length, int)
                or isinstance(max_length, bool)
                or max_length <= 0
            ):
                raise ValueError("max_length must be a positive integer")

        if padding == "max_length" and max_length is None:
            raise ValueError(
                "max_length is required when padding='max_length'"
            )

        if padding in (False, None) and pad_to_multiple_of is not None:
            raise ValueError(
                "pad_to_multiple_of requires padding to be enabled"
            )

        # Batch inputs are raw texts unless explicitly wrapped before calling this method.
        sequences = [
            self.encode(text, chat_wrapped=False)
            for text in texts
        ]

        # Truncate sequences if required
        if truncation:

            for i, sequence in enumerate(sequences):

                if len(sequence) > max_length:

                    if truncation_side == "right":
                        sequences[i] = sequence[:max_length]

                    else:
                        sequences[i] = sequence[-max_length:]

        # Determine target batch length
        if padding in (True, "longest"):
            target_length = max(
                (len(sequence) for sequence in sequences),
                default=0
            )

        elif padding == "max_length":
            target_length = max_length

        else:
            target_length = None

        # Pad to a multiple if requested
        if target_length is not None and pad_to_multiple_of is not None:

            target_length = (
                (target_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
            ) * pad_to_multiple_of

        # Pad sequences and construct attention masks
        input_ids = []
        attention_masks = []

        for sequence in sequences:

            mask = [1] * len(sequence)

            if target_length is not None:

                pad_length = target_length - len(sequence)

                if pad_length < 0:
                    raise ValueError(
                        "A sequence exceeds the padding length. Enable truncation."
                    )

                if pad_length > 0:

                    if self.pad_token_id is None:
                        raise ValueError(
                            "No padding token ID found in tokenizer configuration"
                        )

                    pad_tokens = [self.pad_token_id] * pad_length
                    pad_mask = [0] * pad_length

                    if padding_side == "right":
                        sequence = sequence + pad_tokens
                        mask = mask + pad_mask

                    else:
                        sequence = pad_tokens + sequence
                        mask = pad_mask + mask

            input_ids.append(sequence)
            attention_masks.append(mask)

        result = {
            "input_ids": input_ids
        }

        if return_attention_mask:
            result["attention_mask"] = attention_masks

        return result


# Testing
if __name__ == "__main__":

    tokenizer = Qwen3Tokenizer()

    print("Vocabulary size:", len(tokenizer.vocabulary))
    print("Merge count:", len(tokenizer.merge_ranks))
    print("Added tokens:", len(tokenizer.added_token_to_id))

    # Configuration
    print("\nTokenizer configuration:")
    print("BOS:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("PAD:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("Stop IDs:", tokenizer.stop_token_ids)
    print("Maximum length:", tokenizer.model_max_length)

    # Ordinary text
    text = "Hello, how are you?"

    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)

    print("\nInput:", text)
    print("Token IDs:", token_ids)
    print("Decoded:", decoded_text)

    assert decoded_text == normalize_text(text)

    # Unicode test
    text = "Hello! नमस्ते 世界 😊"

    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)

    assert decoded_text == normalize_text(text)

    # Added token test
    added_token = next(iter(tokenizer.added_token_to_id))

    text = f"Hello {added_token} world"

    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)

    print("\nAdded token test:")
    print("Input:", text)
    print("Token IDs:", token_ids)
    print("Decoded:", decoded_text)

    assert decoded_text == normalize_text(text)

    # Skip special tokens
    print(
        "\nDecoded without special tokens:",
        tokenizer.decode(token_ids, skip_special_tokens=True)
    )

    # Chat formatting
    messages = [
        {"role": "user", "content": "What is reinforcement learning?"}
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False
    )

    chat_token_ids = tokenizer.encode_chat(
        messages,
        add_generation_prompt=True,
        enable_thinking=False
    )

    print("\nFormatted chat:")
    print(formatted_text)

    print("\nChat token IDs:", chat_token_ids)

    assert tokenizer.decode(chat_token_ids) == normalize_text(formatted_text)

    # Previous single-message interface
    chat_ids = tokenizer.encode_chat(
        user_msg="Explain PPO.",
        add_generation_prompt=True,
        enable_thinking=False
    )

    print("\nSingle-message chat IDs:", chat_ids)

    # Batch encoding and padding
    texts = [
        "Hello",
        "How are you?",
        "Explain reinforcement learning."
    ]

    batch = tokenizer.encode_batch(
        texts,
        padding=True,
        truncation=True,
        max_length=128
    )

    print("\nBatch input IDs:", batch["input_ids"])
    print("Attention masks:", batch["attention_mask"])

    lengths = [len(ids) for ids in batch["input_ids"]]

    assert len(batch["input_ids"]) == len(texts)
    assert len(set(lengths)) == 1

    print("\nAll tokenizer tests passed!")
