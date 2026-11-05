# Ollama, llama.cpp, ONNX Runtime, or Candle?

Keep **Ollama as the default** for this app. The local machine reports Apple M1 Max, 64 GB unified memory, and arm64. A direct `llama-server` is worth benchmarking for your main text model, but there is no measured speedup to claim here. Ollama already integrates llama.cpp for supported GGUF models, supports Metal, and also has an MLX path for selected architectures. Which engine runs a particular model matters more than whether the HTTP application is Python or Rust. [Ollama backends](https://github.com/ollama/ollama), [Metal support](https://docs.ollama.com/gpu), [MLX preview](https://ollama.com/blog/mlx).

The local CLI reports version 0.40.11. `llama-server` is already installed at `/opt/homebrew/bin/llama-server` (version 0.7.0, build 11669, commit `620d2225c`). Ollama was stopped even when checked outside the sandbox, and the inspected `~/.ollama/models` directory was empty. `data/list.txt` is therefore an inventory snapshot, not proof that these weights exist on this machine or currently load. No inference benchmark was run and no models were downloaded.

## Use the models you already have

These are starting assignments, not quality rankings:

| Task | Starting model from `data/list.txt` | Notes |
| --- | --- | --- |
| Everyday chat / draft | `qwen2.5:3b` | Keep one general assistant as the default. |
| Code | `qwen2.5-coder:3b` | Select explicitly for code tasks. |
| Agent review | `phi4-mini:latest` | Review the first stage's answer. |
| Agent synthesis | `llama3.2:3b` | Merge the draft and review. |
| Image understanding | `moondream:1.8b` or your WeVisOne tag | Confirm actual image capability with `/api/show`; a GGUF name alone does not establish projector compatibility. |
| Image generation | `x/flux2-klein:4b` | Diffusion/image generation, separate from vision input. |
| Semantic recall | `all-minilm:latest` | Embeddings, separate from chat. |
| Other experiments | Nemotron, Spark-X2.6, LFM3, DeepSeek, Gemma | Test your actual tasks; parameter counts and disk size alone do not establish speed or quality. |

Inspect a model without starting inference:

```sh
curl http://127.0.0.1:11434/api/show \
  -H 'Content-Type: application/json' \
  -d '{"model":"moondream:1.8b"}'
```

Look at `capabilities`, model metadata, and projector information. [Show API](https://docs.ollama.com/api/show).

## One model at a time

Start a manually managed Ollama daemon with these limits, after quitting any existing Ollama server:

```sh
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 ollama serve
```

For the macOS Ollama application, set its environment with `launchctl`, then restart Ollama:

```sh
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
launchctl setenv OLLAMA_NUM_PARALLEL 1
```

The app also needs a shared inference lock: inspect/unload the previous model, run the next operation, wait for unloading, then release the lock. Include embeddings, vision, image generation, and manual memory release in that policy. Use `keep_alive: 0` to unload after a request; an explicit unload request is:

```sh
curl http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:3b","keep_alive":0}'
ollama ps
```

Immediate unloading trades memory for repeated loading time. Embedding then chatting means two sequential loads. Three agent stages mean three loads. Other clients bypassing this app can still make requests; daemon limits help constrain them. Ollama controls do not unload a separately running Python/MLX worker. [Ollama memory and concurrency configuration](https://docs.ollama.com/faq).

## FLUX can run through Ollama without Python

Ollama officially offers experimental macOS image generation, including FLUX.2 Klein. Its official SDK uses `/api/generate`; dimensions and steps are top-level request fields. Streaming responses include progress and an `image` field containing base64 output. [Official announcement](https://ollama.com/blog/image-generation), [SDK example](https://github.com/ollama/ollama-js/blob/main/examples/image-generation/image-generation.ts), [request/response types](https://github.com/ollama/ollama-js/blob/main/src/interfaces.ts).

```json
{
  "model": "x/flux2-klein:4b",
  "prompt": "A ceramic cup on a sunlit wooden table",
  "width": 512,
  "height": 512,
  "steps": 4,
  "stream": true,
  "keep_alive": 0
}
```

The client must parse NDJSON, decode `image`, validate that it is an image, and save it. Treat unsupported runtimes or missing image output as errors. A procedural placeholder is not a successful model generation. Native Ollama weights and a separate mflux model installation are not interchangeable paths.

## Trying llama.cpp separately

The app has a separate `LLM_BACKEND=llamacpp` adapter for local GGUFs. It starts a server for the selected model and stops it after inference, preserving the unload policy. It discovers GGUFs in `data/models/` and `data/` and can reuse compatible models from an existing Ollama cache. An explicit, nonempty registry overrides discovery. Ollama remains the image-generation backend unless `IMAGE_BACKEND=mlx` is explicitly selected. Pointing `OLLAMA_URL` at `llama-server` does not make it implement Ollama's API.

To override discovery, create `data/llamacpp-models.json` with models you already have:

```json
[
  {
    "id": "qwen2.5:3b",
    "name": "Qwen 2.5 3B",
    "path": "/absolute/path/to/model.gguf"
  }
]
```

For a supported vision architecture, also provide `"mmproj": "/absolute/path/to/matching-projector.gguf"`. Paths can be relative to the registry's directory. The adapter checks the GGUF header but cannot establish architecture compatibility until the runtime loads it.

```sh
LLM_BACKEND=llamacpp LLAMACPP_BIN=/absolute/path/to/llama-server \
  cargo run --release
```

Optional settings: `LLAMACPP_MODELS_FILE` (default `data/llamacpp-models.json`), `LLAMACPP_PORT` (11435), `LLAMACPP_CTX_SIZE` (4096), and `LLAMACPP_GPU_LAYERS` (99). The adapter runs one loopback server at a time, waits for health, calls `/v1/chat/completions`, then kills and awaits the process on success or failure. Semantic embedding recall is skipped for this backend. The runtime binary must already be installed.

When direct llama.cpp is selected and no usable local models are found, startup can queue the default Qwen 2.5 3B Q4_K_M download into `data/models/`. Set `AUTO_DOWNLOAD_MODEL=0` to disable startup downloads. The Models view also exposes an allowlisted catalog and transfer progress. Transfers run sequentially, write unique `.part` files, and verify the publisher's SHA-256 plus file size and GGUF header before atomic rename. Failed transfers remain visible and can be retried; partial files are discarded rather than advertised as models. The starter is approximately 2.1 GB. Sources: [Qwen general model](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/blob/main/qwen2.5-3b-instruct-q4_k_m.gguf), [Qwen coder model](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/blob/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf).

The following commands are also useful for a standalone benchmark experiment.

Build outside this app's source tree. Pin and record the tested llama.cpp commit. Metal is enabled by default on macOS. [Build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build --config Release -j 8
git rev-parse HEAD
```

Use a compatible local GGUF, ideally the identical quantized weights used for the Ollama comparison:

```sh
./build/bin/llama-server \
  -m /absolute/path/to/model.gguf \
  --host 127.0.0.1 --port 8081 -c 4096 -np 1 -ngl 99
```

For sequential specialist models, stop and await the previous server process before starting another. A server left running retains its model; Ollama's `keep_alive` parameter does not control it. Vision models generally need a matching `--mmproj` alongside the text GGUF, supported architecture, and correct image preprocessing. `llama.cpp` multimodal input is not a replacement FLUX diffusion engine. [Server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [multimodal setup](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).

## Measure the right things

Compare identical weights, quantization, context length, prompt, output limit, GPU offload, and sampling settings. Record model/runtime versions, memory pressure, cold start, warm generation, and complete three-stage agent time. Run one engine at a time and repeat at least five times. Use your OCR documents, coding prompts, and image prompts as a quality check alongside timing.

A repeatable, standard-library-only developer utility is provided at `scripts/benchmark.py`. It is optional; the application does not require Python. Start only the engine being tested and use identical weights/quantization. For llama.cpp, start the server with `-c 4096 -np 1`; the script verifies the context through `/props`.

```sh
python3 scripts/benchmark.py --backend ollama --model qwen2.5:3b \
  --mode warm --runs 5 --output /tmp/ollama-warm.json

python3 scripts/benchmark.py --backend llamacpp --model qwen2.5:3b \
  --mode warm --runs 5 --output /tmp/llamacpp-warm.json

python3 scripts/benchmark.py --backend ollama --model qwen2.5:3b \
  --mode cold --runs 5 --output /tmp/ollama-cold.json
```

Use the same `--prompt-file /path/to/prompt.txt` for real prompts. Reports include wall time, generated token count, decode speed, and generated text. The changing deterministic sample prefix reduces prompt-cache reuse. Ollama cold mode unloads before each measured call; warm mode excludes one warm-up and unloads after the run. llama.cpp results measure an already running server and exclude startup, so comparing them to Ollama cold times would be misleading. This utility intentionally rejects llama.cpp cold mode; measure the app's per-request process lifecycle separately for that case. `--dry-run` validates configuration without inference.

Ollama returns nanosecond durations. Decode throughput is `eval_count * 1e9 / eval_duration`; loading time is `load_duration / 1e9`. For warm measurements use a short keep-alive temporarily, repeat while loaded, and unload afterward. For cold measurements unload before each run. Capture client wall time separately. [Generate API metrics](https://docs.ollama.com/api/generate).

```sh
curl http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -o /tmp/ferro64-ollama-bench.json -w 'wall_seconds=%{time_total}\n' \
  -d '{"model":"qwen2.5:3b","prompt":"Explain a Rust ownership example.","raw":true,"stream":false,"keep_alive":"5m","options":{"temperature":0,"num_ctx":4096,"num_predict":128}}'

jq '{load_seconds:(.load_duration/1e9), prompt_tokens:.prompt_eval_count, output_tokens:.eval_count, decode_tokens_per_second:(if .eval_duration > 0 then .eval_count*1e9/.eval_duration else null end)}' \
  /tmp/ferro64-ollama-bench.json
```

Run the same raw prompt through llama-server's `/completion` endpoint with `n_predict:128`, `temperature:0`, `cache_prompt:false`, and `stream:false`; capture wall time and its timing fields. Raw prompts remove chat-template differences for an engine comparison. Also test normal chat templates for actual application behavior.

For a compute-focused baseline:

```sh
./build/bin/llama-bench -m /absolute/path/to/model.gguf \
  -p 512 -n 128 -ngl 99 -r 5 -o json
```

`llama-bench` separates prompt processing and token generation, but excludes tokenization and sampling time; it is not an end-to-end chat latency measurement. [Benchmark documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md).

Choose direct llama.cpp only if your measurements show a useful improvement that outweighs model/projector management. Rust/Actix improves the application's organization and request handling; it cannot by itself accelerate GPU inference.

## Would ONNX Runtime or Candle be faster?

There is no reliable ranking without measuring the same model and workload on this M1 Max. My recommendation is to compare **Ollama and direct llama.cpp first** for the existing GGUF collection. Consider ONNX Runtime for a specific OCR/embedding model with a supported ONNX export, or Candle when embedding a supported model directly into Rust is a product requirement. This is an engineering recommendation, not a measured performance result.

| Option | Practical fit | Work needed for this app |
| --- | --- | --- |
| Ollama | Existing mixed model collection, discovery, unloading, native macOS FLUX | Already integrated; benchmark the installed runtime and models. |
| Direct llama.cpp | Local GGUF text and supported multimodal models | Model/projector registry and process lifecycle; the app includes an optional adapter. |
| ONNX Runtime | A fixed ONNX model with an effective hardware execution provider | Export/obtain compatible ONNX weights, implement preprocessing and tokenization, select execution provider, and profile graph placement. |
| Candle | In-process Rust inference for a supported architecture | Integrate model-specific code, weights/config/tokenizer, generation loop, and memory lifecycle. |

ONNX is a model format; ONNX Runtime executes it. It does not directly run an arbitrary Ollama tag or GGUF. Its GenAI layer supplies token generation, sampling, and KV-cache management for supported ONNX models. [ONNX Runtime GenAI](https://onnxruntime.ai/docs/genai/).

On Apple hardware, the CoreML execution provider can use CPU, GPU, and Neural Engine, but operator support, input shapes, and actual graph placement matter. Dynamic shapes can reduce performance, and enabling the provider does not guarantee that the entire model runs on the Neural Engine. Quantization also does not guarantee acceleration; its overhead can outweigh savings on some hardware. Benchmark rather than assuming an ONNX export will improve a decoder LLM. [CoreML execution provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html), [quantization performance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

Candle is a Rust ML framework with model implementations and a Metal feature. Its examples include quantized language models, Moondream, TrOCR, embeddings, and diffusion models. That makes it a possible Python-free route for selected architectures, but not an automatic loader for every model in this inventory. Verify the exact architecture, quantization, tensor layout, and image projector before choosing it. Rust ownership of the web server does not make Rust model kernels inherently faster than llama.cpp or MLX kernels. [Candle models](https://github.com/huggingface/candle), [Metal feature](https://github.com/huggingface/candle/blob/main/candle-core/Cargo.toml).

Strict unloading increases time to first token because the next operation reloads model weights and rebuilds its inference state. Include that overhead in the benchmark. A faster warm kernel can still produce a slower three-model workflow if its cold loads are expensive.
