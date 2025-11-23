import argparse, base64, gc, inspect, json, os, sys, tempfile, shutil
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

from tts_playground.runtime.idle_timer import DebouncedIdle


class AdapterBox:
    def __init__(self, module: str, cls_name: str, idle_secs: int, exit_on_idle: bool):
        self.module = module
        self.cls_name = cls_name
        self.exit_on_idle = exit_on_idle
        self.idle = DebouncedIdle(self._on_idle, idle_secs=idle_secs)
        self.inst = None
        self.init_sig: Tuple[Tuple[str, Any], ...] | None = None
        self.load_sig: Tuple[Tuple[str, Any], ...] | None = None
        self.last_init: Dict[str, Any] = {}
        self._clone_tmp: Optional[Path] = None
        self._loaded = False

    def _on_idle(self):
        self._unload()
        if self.exit_on_idle:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    def _unload(self):
        if self.inst is not None:
            try:
                import torch
                # dev = getattr(self.inst, "device", None)
                del self.inst
                self.inst = None
                self._loaded = False
                self.init_sig = None
                self.load_sig = None
                gc.collect()
                try: torch.cuda.synchronize()
                except Exception: pass
                try: torch.cuda.empty_cache()
                except Exception: pass
                try: torch.cuda.ipc_collect()
                except Exception: pass
                if self._clone_tmp:
                    shutil.rmtree(self._clone_tmp, ignore_errors=True)
                    self._clone_tmp = None
            except Exception:
                pass

    def ensure_inited(self, init_kwargs: Optional[Dict[str, Any]] = None):
        if init_kwargs is None:
            if self.inst is not None:
                self.idle.reset()
                return self.inst
            init_kwargs = self.last_init
        sig = tuple(sorted((init_kwargs or {}).items()))
        if self.inst is None or self.init_sig != sig:
            self._unload()
            mod = __import__(self.module, fromlist=[self.cls_name])
            cls = getattr(mod, self.cls_name)
            self.inst = cls(**(init_kwargs or {}))
            self.init_sig = sig
            self.last_init = init_kwargs or {}
        self.idle.reset()
        return self.inst

    def load_model_call(self, inst: Any, load_kws: Dict[str, Any]):
        fn = getattr(inst, "load_model", None)
        if not fn:
            return
        sig = inspect.signature(fn)
        fn(**{k: v for k, v in load_kws.items() if k in sig.parameters})

    def call_clone(self, inst: Any, clone: Dict[str, Any]):
        fn = getattr(inst, "clone_voice", None)
        if not fn:
            return None
        # For adapters like IndexTTS2 that expect single str for ref_audio, take first if list
        if "ref_audio" in clone and isinstance(clone["ref_audio"], list) and len(clone["ref_audio"]) > 0:
            clone = clone.copy()
            clone["ref_audio"] = clone["ref_audio"][0]
        sig = inspect.signature(fn)
        return fn(**{k: v for k, v in clone.items() if k in sig.parameters})

    def call_synth(self, inst: Any, *, text: str, synth_kwargs: Dict[str, Any]) -> tuple[bytes, int]:
        fn = getattr(inst, "synthesize")
        #sig = inspect.signature(fn)
        #wav_bytes = fn(text=text, **{k: v for k, v in synth_kwargs.items() if k in sig.parameters})
        wav_bytes = fn(text=text, **synth_kwargs)
        sr = getattr(inst, "sr", None) or 24000
        self.idle.reset()
        return wav_bytes, int(sr)


class JsonLoop:
    def __init__(self, box: AdapterBox, adapter: str):
        self.box = box
        self.adapter = adapter  # NEW: Store it here

    # ---------- blob staging helpers ---------- (unchanged)
    @staticmethod
    def _is_blob(x):
        return isinstance(x, dict) and "b64" in x

    @staticmethod
    def _safe_name(name: str, fallback: str) -> str:
        name = (name or fallback).strip().replace("\\", "/").split("/")[-1]
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)

    def _stage_blob(self, spec, dstdir: Path, default_name: str) -> str:
        dstdir.mkdir(parents=True, exist_ok=True)
        fname = self._safe_name(spec.get("name", ""), default_name)
        path = dstdir / fname
        path.write_bytes(base64.b64decode(spec["b64"]))
        return str(path)

    def _stage_list(self, items, dstdir: Path, base: str):
        out = []
        for i, it in enumerate(items or []):
            out.append(self._stage_blob(it, dstdir, f"{base}_{i}.wav") if self._is_blob(it) else it)
        return out

    def _stage_map(self, mp, dstdir: Path, base: str):
        out = {}
        for k, v in (mp or {}).items():
            sid = str(int(k))
            out[sid] = self._stage_blob(v, dstdir, f"{base}_{sid}.wav") if self._is_blob(v) else v
        return out

    def _prepare_clone(self, clone: dict | None, tmpdir: Path) -> dict:
        if clone is None:
            clone = {}
        c = dict(clone)
        if "ref_audio" in c:
            ra = c["ref_audio"]
            if self._is_blob(ra):
                c["ref_audio"] = self._stage_blob(ra, tmpdir, "ref.wav")
            elif isinstance(ra, list):
                c["ref_audio"] = self._stage_list(ra, tmpdir, "ref")
        if "voice_samples" in c:
            c["voice_samples"] = self._stage_list(c["voice_samples"], tmpdir, "ref")
        if "speaker_voices" in c:
            c["speaker_voices"] = self._stage_map(c["speaker_voices"], tmpdir, "spk")
        return c

    def _prepare_args(self, args: dict | None, tmpdir: Path) -> dict:
        if args is None:
            args = {}
        a = dict(args)
        if "voice_samples" in a:
            a["voice_samples"] = self._stage_list(a["voice_samples"], tmpdir, "ref")
        if "speaker_voices" in a:
            a["speaker_voices"] = self._stage_map(a["speaker_voices"], tmpdir, "spk")
        # Handle IndexTTS2 emo_audio_prompt
        if "emo_audio_prompt" in a and self._is_blob(a["emo_audio_prompt"]):
            a["emo_audio_prompt"] = self._stage_blob(a["emo_audio_prompt"], tmpdir, "emo.wav")
        return a

    def run(self):
        for line in sys.stdin:
            line = line.rstrip('\r\n')
            sys.stdout.write(f"[DEBUG INPUT {self.adapter}] Raw input line (len={len(line)}): {repr(line[:100])}{'...' if len(line) > 100 else ''}\n")
            sys.stdout.flush()
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"[DEBUG INPUTERR] JSON decode failed: {str(e)} on {repr(line)}\n")
                sys.stderr.flush()
                continue
            rid = msg.get("id")
            method = msg.get("method")
            params = msg.get("params", {})
            sys.stdout.write(f"[DEBUG PARSED {self.adapter}] rid={rid}, method={method}\n")
            sys.stdout.flush()
            try:
                if method == "run":
                    sys.stdout.write(f"[DEBUG START RUN {self.adapter}] Processing run with params keys: {list(params.keys())}\n")  
                    sys.stdout.flush()
                    init = params.get("init", {})
                    load_kws = params.get("load_model", {})
                    clone_dict = params.get("clone_voice", {})
                    text = params["synthesize"]["text"]
                    synth_kws = params["synthesize"].get("kwargs", {})
                    old_tmp = self.box._clone_tmp
                    tmpdir_clone = Path(tempfile.mkdtemp(prefix="tts_clone_"))
                    tmpdir_synth = Path(tempfile.mkdtemp(prefix="tts_synth_"))
                    try:
                        sys.stdout.write(f"[DEBUG ENSURE {self.adapter}] Calling ensure_inited\n")  
                        sys.stdout.flush()
                        inst = self.box.ensure_inited(init)
                        sys.stdout.write(f"[DEBUG ENSURE OK {self.adapter}] Instance ready\n")  
                        sys.stdout.flush()
                        # Load model if sig changed
                        load_sig = tuple(sorted(load_kws.items()))
                        if self.box.load_sig != load_sig:
                            sys.stdout.write(f"[DEBUG LOAD {self.adapter}] Calling load_model\n")  
                            sys.stdout.flush()
                            try:
                                self.box.load_model_call(inst, load_kws)
                                sys.stdout.write(f"[DEBUG LOAD SUCCESS {self.adapter}] No exception in load_model_call\n")
                                sys.stdout.flush()
                            except Exception as load_e:
                                import traceback
                                sys.stdout.write(f"[DEBUG LOAD EXCEPT {self.adapter}] Exception in load_model_call: {str(load_e)}\n")
                                sys.stdout.write(f"[DEBUG LOAD TRACE {self.adapter}] {traceback.format_exc()}\n")
                                sys.stdout.flush()
                                raise  # Re-raise to hit outer except
                            self.box.load_sig = load_sig
                            self.box._loaded = True
                            sys.stdout.write(f"[DEBUG LOAD OK {self.adapter}] Model loaded\n")
                        self.box.idle.reset()
                        # Clone
                        clone = self._prepare_clone(clone_dict, tmpdir_clone)
                        sys.stdout.write(f"[DEBUG CLONE {self.adapter}] Prepared clone, calling clone_voice\n")  
                        sys.stdout.flush()
                        self.box.call_clone(inst, clone)
                        self.box._clone_tmp = tmpdir_clone
                        if old_tmp:
                            shutil.rmtree(old_tmp, ignore_errors=True)
                        self.box.idle.reset()
                        sys.stdout.write(f"[DEBUG CLONE OK {self.adapter}] Voice cloned\n")  
                        sys.stdout.flush()
                        # Synth
                        synth_kws = self._prepare_args(synth_kws, tmpdir_synth)
                        sys.stdout.write(f"[DEBUG SYNTH {self.adapter}] Prepared synth kwargs, text len={len(text)}, calling synthesize\n")  
                        sys.stdout.flush()
                        wav, sr = self.box.call_synth(inst, text=text, synth_kwargs=synth_kws)
                        sys.stdout.write(f"[DEBUG SYNTH OK {self.adapter}] Got wav len={len(wav)}, sr={sr}\n")  
                        sys.stdout.flush()
                        b64 = base64.b64encode(wav).decode("ascii")
                        out = {"ok": True, "result": {"wav_b64": b64, "sr": sr}}
                    finally:
                        shutil.rmtree(tmpdir_synth, ignore_errors=True)
                else:
                    out = {"ok": False, "error": f"unknown method: {method}"}
            except Exception as e:
                import traceback
                sys.stdout.write(f"[DEBUG EXCEPT {self.adapter}] Caught exception: {str(e)}\n")  
                sys.stdout.write(f"[DEBUG TRACE {self.adapter}] Traceback:\n{traceback.format_exc()}\n")  
                sys.stdout.flush()
                out = {"ok": False, "error": str(e)}
            out["id"] = rid
            sys.stdout.write(f"[DEBUG WRITE {self.adapter}] Writing response for rid={rid}, ok={out['ok']}\n")  
            sys.stdout.flush()
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
            sys.stdout.write(f"[DEBUG WRITE OK {self.adapter}] Flushed response\n")  
            sys.stdout.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--module", required=True)
    ap.add_argument("--cls", required=True)
    ap.add_argument("--idle", type=int, default=180)
    ap.add_argument("--exit-on-idle", type=int, default=0)
    args = ap.parse_args()

    box = AdapterBox(args.module, args.cls, idle_secs=args.idle, exit_on_idle=bool(args.exit_on_idle))
    JsonLoop(box, args.adapter).run()