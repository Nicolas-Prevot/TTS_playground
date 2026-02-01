import os
import copy
from typing import List, Optional

import torch
import langid
import jieba
import tqdm
from loguru import logger
from dataclasses import asdict
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache

from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

from .audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from tts_core.base import BaseTTS


def normalize_chinese_punctuation(text: str) -> str:
    """Converts Chinese (full-width) punctuation marks to English (half-width) equivalents."""
    chinese_to_english_punct = {
        "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!", "（": "(", "）": ")",
        "【": "[", "】": "]", "《": "<", "》": ">", "“": '"', "”": '"', "‘": "'", "’": "'",
        "、": ",", "—": "-", "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text


def prepare_chunk_text(
    text,
    chunk_method: Optional[str] = None,
    chunk_max_word_num: int = 100,
    chunk_max_num_turns: int = 1,
):
    """Chunk the text into smaller pieces. We will later feed the chunks one by one to the model."""
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i: i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for _, paragraph in enumerate(paragraphs):
            if language == "zh":
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i: i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i: i + chunk_max_word_num])
                    chunks.append(chunk)
            chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def _resolve_device(requested: Optional[str]) -> str:
    """Resolve requested device into a usable runtime device."""
    req = (requested or "").strip().lower()
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return req if ":" in req else "cuda:0"
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    if req == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS requested but not available. Falling back to CPU.")
        return "cpu"
    if req in ("cpu", ""):
        return "cpu"
    logger.warning("Unknown device '{}'. Falling back to CPU.", req)
    return "cpu"


def _choose_dtype(device: str) -> torch.dtype:
    """Pick a safe dtype for the chosen device."""
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


class HiggsAudioModelClient:
    """Encapsulates Higgs Audio model loading and inference."""

    def __init__(
        self,
        model_path: str,
        audio_tokenizer,
        device: str = "cpu",
        max_new_tokens: int = 2048,
        kv_cache_lengths: Optional[List[int]] = None,
        use_static_kv_cache: bool = False,
    ):
        self._device = _resolve_device(device)
        self._dtype = _choose_dtype(self._device)

        self._audio_tokenizer = audio_tokenizer
        self._model = HiggsAudioModel.from_pretrained(
            model_path,
            device_map=self._device,
            torch_dtype=self._dtype,
        )
        self._model.eval()

        self._kv_cache_lengths = kv_cache_lengths or [1024, 4096, 8192]
        self._use_static_kv_cache = bool(use_static_kv_cache and self._device.startswith("cuda"))

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
        self._max_new_tokens = max_new_tokens
        self._collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self._config.audio_in_token_idx,
            audio_out_token_id=self._config.audio_out_token_idx,
            audio_stream_bos_id=self._config.audio_stream_bos_id,
            audio_stream_eos_id=self._config.audio_stream_eos_id,
            encode_whisper_embed=self._config.encode_whisper_embed,
            pad_token_id=self._config.pad_token_id,
            return_audio_in_tokens=self._config.encode_audio_in_tokens,
            use_delay_pattern=self._config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self._config.audio_num_codebooks,
        )

        self.kv_caches = None
        if self._use_static_kv_cache:
            self._init_static_kv_cache()

    def _init_static_kv_cache(self):
        cache_config = copy.deepcopy(self._model.config.text_config)
        cache_config.num_hidden_layers = self._model.config.text_config.num_hidden_layers
        if self._model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self._model.config.audio_dual_ffn_layers)

        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self._model.device,
                dtype=self._model.dtype,
            )
            for length in sorted(self._kv_cache_lengths)
        }

        if "cuda" in self._device:
            logger.info("Capturing CUDA graphs for each KV cache length")
            self._model.capture_model(self.kv_caches.values())

    def _prepare_kv_caches(self):
        if self.kv_caches:
            for kv_cache in self.kv_caches.values():
                kv_cache.reset()

    @torch.inference_mode()
    def generate(
        self,
        messages,
        audio_ids,
        chunked_text,
        generation_chunk_buffer_size,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        seed=123,
    ):
        sr = 24000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []

        for idx, chunk_text in tqdm.tqdm(
            enumerate(chunked_text), desc="Generating audio chunks", total=len(chunked_text)
        ):
            generation_messages.append(Message(role="user", content=chunk_text))
            chatml_sample = ChatMLSample(messages=messages + generation_messages)
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self._tokenizer)

            postfix = self._tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
                add_special_tokens=False
            )
            input_tokens.extend(postfix)

            logger.debug(f"========= Chunk {idx} Input =========")
            logger.debug(self._tokenizer.decode(input_tokens))

            context_audio_ids = audio_ids + generated_audio_ids

            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1)
                if context_audio_ids
                else None,
                audio_ids_start=torch.cumsum(
                    torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0
                )
                if context_audio_ids
                else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            batch_data = self._collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)

            if self._use_static_kv_cache:
                self._prepare_kv_caches()

            gen_kwargs = dict(
                **batch,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self._tokenizer,
                seed=seed,
            )
            if self.kv_caches is not None:
                gen_kwargs["past_key_values_buckets"] = self.kv_caches

            outputs = self._model.generate(**gen_kwargs)

            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self._config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                step_audio_out_ids_l.append(
                    audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1]
                )

            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)

            generation_messages.append(Message(role="assistant", content=AudioContent(audio_url="")))

            if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
                generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
                generation_messages = generation_messages[(-2 * generation_chunk_buffer_size):]

        logger.debug("========= Final Text output =========")
        logger.debug(self._tokenizer.decode(outputs[0][0]))

        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)
        input_ids = concat_audio_out_ids.unsqueeze(0).to(self._device)
        concat_wv = self._audio_tokenizer.decode(input_ids)[0, 0]

        text_result = self._tokenizer.decode(outputs[0][0])
        return concat_wv, sr, text_result


class HiggsAudioAdapter(BaseTTS):
    """
    Adapter for Boson AI's Higgs Audio for local text-to-speech synthesis.
    This model supports zero-shot and one-shot voice cloning.
    """

    def __init__(
        self,
        *,
        model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
        device: str = None,
        use_static_kv_cache: bool = True,
        max_new_tokens: int = 4096,
    ):
        super().__init__()
        self.model_path = model_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_static_kv_cache = use_static_kv_cache
        self.max_new_tokens = max_new_tokens

        self.model_client: Optional[HiggsAudioModelClient] = None
        self.sr = 24000

        self.ref_audio: Optional[str] = None
        self.ref_text: Optional[str] = None
        self.scene_prompt: Optional[str] = None
        self.messages: List[Message] = []
        self.audio_ids: List[torch.Tensor] = []
        self._context_prepared: bool = False

    def load_model(self):
        """Loads the Higgs Audio model, text tokenizer, and audio tokenizer."""
        resolved = _resolve_device(self.device)
        logger.info(f"Loading Higgs Audio model on device: {resolved} (requested: {self.device})")

        audio_tokenizer = load_higgs_audio_tokenizer(self.audio_tokenizer_path, device=resolved)

        self.model_client = HiggsAudioModelClient(
            model_path=self.model_path,
            audio_tokenizer=audio_tokenizer,
            device=resolved,
            max_new_tokens=self.max_new_tokens,
            use_static_kv_cache=self.use_static_kv_cache,
        )
        self.sr = 24000
        logger.info("Higgs Audio model loaded successfully. 🚀")

    def _prepare_context(self):
        """Prepares the generation context from reference materials."""
        if self.model_client is None:
            raise RuntimeError("Model must be loaded before preparing context.")

        messages: List[Message] = []
        audio_ids: List[torch.Tensor] = []

        scene = self.scene_prompt or "Audio is recorded from a quiet room."
        system_content = (
            "Generate audio following instruction.\n"
            f"<|scene_desc_start|>\n{scene}\n<|scene_desc_end|>\n"
            "<|speaker_id_start|>SPEAKER0<|speaker_id_end|>"
        )
        messages.append(Message(role="system", content=system_content))

        if self.ref_audio and self.ref_text:
            if not os.path.exists(self.ref_audio):
                raise FileNotFoundError(f"Reference audio file not found: {self.ref_audio}")

            audio_tokens = self.model_client._audio_tokenizer.encode(self.ref_audio)
            audio_ids.append(audio_tokens)
            messages.append(Message(role="user", content=self.ref_text))
            messages.append(Message(role="assistant", content=AudioContent(audio_url=self.ref_audio)))

        self.messages = messages
        self.audio_ids = [aid.to(_resolve_device(self.device)) for aid in audio_ids]
        self._context_prepared = True

    def clone_voice(self, ref_audio: str = None, ref_text: str = None, *, scene_prompt: str = None):
        """Set reference audio + transcript (and optional scene prompt) for voice cloning."""
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.scene_prompt = scene_prompt
        self._context_prepared = False
        return True

    def synthesize(
        self,
        text: str,
        *,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        ras_win_len: int = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
        chunk_method: str | None = None,
        chunk_max_word_num: int = 200,
        generation_chunk_buffer_size: Optional[int] = None,
    ) -> bytes:
        """Generate audio bytes (WAV) from input text."""
        if self.model_client is None:
            raise RuntimeError("Model not loaded; call load_model() first.")

        if not self._context_prepared:
            self._prepare_context()

        normalized_text = normalize_chinese_punctuation(text)
        chunked_text = prepare_chunk_text(
            text=normalized_text,
            chunk_method=chunk_method,
            chunk_max_word_num=chunk_max_word_num,
        )

        logger.info("Messages: {}", self.messages)

        wav_np, sr, _ = self.model_client.generate(
            messages=self.messages,
            audio_ids=self.audio_ids,
            chunked_text=chunked_text,
            generation_chunk_buffer_size=generation_chunk_buffer_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            ras_win_len=ras_win_len,
            ras_win_max_num_repeat=ras_win_max_num_repeat,
            seed=seed or torch.randint(0, 10000, (1,)).item(),
        )

        return self._wav_to_bytes(wav_np, sr)


if __name__ == "__main__":
    tts = HiggsAudioAdapter()
    tts.load_model()

    tts.clone_voice(
        ref_audio="data/ref/basic_ref_en.wav",
        ref_text="Some call me nature, others call me mother nature.",
        scene_prompt="A clear voice speaking in a quiet room."
    )

    audio_bytes = tts.synthesize(
        text="Hello world! This audio was generated using a cloned voice.",
        temperature=0.95,
        top_p=0.9,
        seed=42
    )

    os.makedirs("data/gen", exist_ok=True)
    with open("data/gen/test.wav", "wb") as f:
        f.write(audio_bytes)
