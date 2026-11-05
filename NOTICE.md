# Notices

Ferro64 is licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

It includes or builds on the work below. Each keeps its own license.

## Included in this repository

| Work | License | Where |
| --- | --- | --- |
| Lumen theme from [Hallmark](https://github.com/Nutlope/hallmark) ([usehallmark.com](https://www.usehallmark.com)) | MIT | `static/styles.css` |
| [Instrument Serif](https://github.com/Instrument/instrument-serif), [Geist and Geist Mono](https://github.com/vercel/geist-font) | SIL Open Font License 1.1 | `static/fonts/` (with `OFL.txt`) |
| [xterm.js](https://xtermjs.org) 6.0.0 and its fit add-on 0.11.0 | MIT | `static/vendor/xterm/` (with `LICENSE`) |

## Downloaded or built on your machine, not included

| Work | License | How |
| --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) | MIT | `scripts/setup.sh` or Homebrew |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (`sd-cli`) | MIT | `scripts/setup.sh` |
| [ltx-video-swift-mlx](https://github.com/VincentGourbin/ltx-video-swift-mlx) | MIT | `scripts/setup-video.sh` builds a pinned release |
| Rust crates (Actix Web, Tera, rusqlite, Tauri, portable-pty and others) | per crate, see `Cargo.lock` | `cargo build` |

## Models

No model weights are included. The app downloads them from Hugging Face when you choose to, each under its publisher's license (for example Apache-2.0 for Qwen, the LTX-2 Community License for LTX-2.3, and the FLUX.2 Klein license for FLUX.2). Check a model's license before using its output commercially.
