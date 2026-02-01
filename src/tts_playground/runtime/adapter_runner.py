import sys
import os
import json
import argparse
import base64
import traceback
import importlib
import threading

class AdapterRunner:
    def __init__(self, module_name, class_name, idle_secs, exit_on_idle):
        self.module_name = module_name
        self.class_name = class_name
        self.idle_secs = idle_secs
        self.exit_on_idle = exit_on_idle
        
        # State tracking
        self.instance = None
        self.last_init_sig = None
        self.last_load_sig = None
        self.cls = None
        
        # Timer setup
        self.timer = None
        self._refresh_timer()
        self._load_adapter_class()

    def _timeout(self):
        # This runs in a separate thread when the timer expires
        sys.stderr.write(f"[Runner] Idle for {self.idle_secs}s. Exiting.\n")
        os._exit(0)

    def _refresh_timer(self):
        if self.timer: 
            self.timer.cancel()
        if self.exit_on_idle and self.idle_secs > 0:
            self.timer = threading.Timer(self.idle_secs, self._timeout)
            self.timer.daemon = True
            self.timer.start()

    def _load_adapter_class(self):
        try:
            mod = importlib.import_module(self.module_name)
            self.cls = getattr(mod, self.class_name)
        except Exception:
            traceback.print_exc()
            sys.exit(1)

    def _get_sig(self, args: dict):
        """Create a hashable signature from dictionary arguments."""
        # Sorting ensures that {"a": 1, "b": 2} is the same signature as {"b": 2, "a": 1}
        return json.dumps(args, sort_keys=True)

    def loop(self):
        # sys.stdin is iterable; iterating it blocks until a line appears
        for line in sys.stdin:
            # Activity detected -> Reset the idle timer
            self._refresh_timer()
            
            line = line.strip()
            if not line: continue
            
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue

            rid = req.get("id")
            method = req.get("method")
            params = req.get("params", {})
            
            resp = {"id": rid, "ok": True, "result": None}
            
            try:
                if method == "run":
                    self._handle_run(params, resp)
                else:
                    raise ValueError(f"Unknown method {method}")
            except Exception as e:
                resp["ok"] = False
                resp["error"] = str(e)
                # Print trace to stderr so it shows up in Docker logs but doesn't break the JSON stdout protocol
                sys.stderr.write(f"[Runner Error] {e}\n")
                traceback.print_exc(file=sys.stderr)

            # Write response to stdout (RunnerManager reads this)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

    def _handle_run(self, params, resp):
        # --- 1. Initialization (Idempotent) ---
        init_kws = params.get("init", {})
        current_init_sig = self._get_sig(init_kws)

        # Only re-instantiate if the class hasn't been created OR arguments changed
        if self.instance is None or current_init_sig != self.last_init_sig:
            sys.stderr.write(f"[Runner] Initializing {self.class_name}...\n")
            if self.instance is not None:
                # Optional: explicit cleanup if the adapter supports it
                del self.instance
                import gc
                gc.collect()
            
            self.instance = self.cls(**init_kws)
            self.last_init_sig = current_init_sig
            self.last_load_sig = None # Reset load state because it's a new instance

        # --- 2. Load Model (Idempotent) ---
        load_kws = params.get("load_model", {})
        current_load_sig = self._get_sig(load_kws)

        # Only call load_model if it exists AND arguments changed (or never loaded)
        if hasattr(self.instance, "load_model"):
            if current_load_sig != self.last_load_sig:
                sys.stderr.write(f"[Runner] Loading model...\n")
                self.instance.load_model(**load_kws)
                self.last_load_sig = current_load_sig
            else:
                # Fast path: Model already loaded with these args
                pass

        # --- 3. Clone Voice ---
        clone_kws = params.get("clone_voice", {})
        if clone_kws and hasattr(self.instance, "clone_voice"):
            self.instance.clone_voice(**clone_kws)

        # --- 4. Synthesize ---
        synth_data = params.get("synthesize", {})
        text = synth_data.get("text")
        kwargs = synth_data.get("kwargs", {})
        
        audio_bytes = self.instance.synthesize(text, **kwargs)
        
        if not isinstance(audio_bytes, bytes):
            raise ValueError(f"Adapter returned {type(audio_bytes)}, expected bytes")

        b64 = base64.b64encode(audio_bytes).decode("ascii")
        # Get sample rate safely
        sr = getattr(self.instance, "sr", 24000)
        
        resp["result"] = {"wav_b64": b64, "sr": sr}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--cls", required=True)
    parser.add_argument("--idle", type=int, default=180)
    parser.add_argument("--exit-on-idle", type=str, default="1")
    args = parser.parse_args()

    runner = AdapterRunner(
        args.module, 
        args.cls, 
        args.idle, 
        args.exit_on_idle == "1"
    )
    sys.stderr.write(f"[Runner] Started {args.cls}. Idle timeout: {args.idle}s\n")
    runner.loop()