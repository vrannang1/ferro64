#!/usr/bin/env bash
# One-time setup for Ferro64, after `git clone`. Safe to run again.
#
#   ./scripts/setup.sh              Rust, the engines, the app, and what fits this machine
#   ./scripts/setup.sh --video      also the video engine (Mac with 32 GB+, needs Xcode)
#   ./scripts/setup.sh --desktop    also the desktop app (macOS: dist/Ferro64.app)
# Windows: scripts\setup.ps1 does the same.
#
# Installs what is missing:
#   Rust                            via rustup, if `cargo` is not found
#   llama-server (llama.cpp)        chat, code, agents
#   sd-cli (stable-diffusion.cpp)   images
#   the OCR helper (macOS)          documents
# It downloads no models: Settings → Models shows which ones fit this machine.
# Override the pinned engine versions with LLAMA_TAG / SD_TAG.
set -euo pipefail

LLAMA_TAG="${LLAMA_TAG:-b12032}"
SD_TAG="${SD_TAG:-master-1039-290148d}"
SD_COMMIT="${SD_TAG##*-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/data/bin"
OS="$(uname -s)"
ARCH="$(uname -m)"
WANT_VIDEO=0
WANT_DESKTOP=0
for arg in "$@"; do
  case "$arg" in
    --video) WANT_VIDEO=1 ;;
    --desktop) WANT_DESKTOP=1 ;;
    -h|--help) sed -n 2,15p "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 1 ;;
  esac
done

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }

# Downloads an archive and flattens the folder holding $3 into $2.
fetch() {
  local url="$1" dest="$2" want="$3" tmp
  tmp="$(mktemp -d)"
  note "downloading $(basename "$url")"
  curl -fL --progress-bar -o "$tmp/archive" "$url"
  mkdir -p "$tmp/x"
  case "$url" in
    *.zip) unzip -q "$tmp/archive" -d "$tmp/x" ;;
    *) tar -xzf "$tmp/archive" -C "$tmp/x" ;;
  esac
  local found
  found="$(find "$tmp/x" -name "$want" -type f | head -1)"
  [ -n "$found" ] || { echo "error: $want not found in $url" >&2; exit 1; }
  rm -rf "$dest" && mkdir -p "$dest"
  cp -R "$(dirname "$found")/." "$dest/"
  chmod +x "$dest/$want"
  rm -rf "$tmp"
}

gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then echo cuda
  elif command -v vulkaninfo >/dev/null 2>&1; then echo vulkan
  else echo cpu; fi
}

# ---------------------------------------------------------------- toolchain
say "1/5 toolchain"
if [ "$OS" = Darwin ]; then
  [ "$ARCH" = arm64 ] || { echo "error: Ferro64 needs an Apple Silicon Mac (M1 or newer)" >&2; exit 1; }
  if ! xcode-select -p >/dev/null 2>&1; then
    note "Apple's command line tools are needed to build. A window opens to install them;"
    note "when it finishes, run this script again."
    xcode-select --install || true
    exit 1
  fi
  note "command line tools: $(xcode-select -p)"
elif [ "$OS" != Linux ]; then
  echo "error: unsupported system $OS (macOS and Linux only)" >&2; exit 1
fi
if ! command -v cargo >/dev/null 2>&1 && [ -x "$HOME/.cargo/bin/cargo" ]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
if ! command -v cargo >/dev/null 2>&1; then
  note "installing Rust with rustup (https://rustup.rs)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
fi
note "$(cargo --version)"

# ----------------------------------------------------------------- engines
say "2/5 engines"
case "$OS" in
  Darwin)
    if command -v llama-server >/dev/null 2>&1; then
      note "llama.cpp: $(llama-server --version 2>&1 | head -1)"
    elif [ -x "$BIN/llamacpp/llama-server" ]; then
      note "llama.cpp: data/bin/llamacpp"
    elif command -v brew >/dev/null 2>&1; then
      note "installing llama.cpp with Homebrew"
      brew install llama.cpp
    else
      fetch "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/llama-$LLAMA_TAG-bin-macos-arm64.tar.gz" "$BIN/llamacpp" llama-server
    fi
    if [ -x "$BIN/sdcpp/sd-cli" ]; then
      note "stable-diffusion.cpp: data/bin/sdcpp"
    else
      fetch "https://github.com/leejet/stable-diffusion.cpp/releases/download/$SD_TAG/sd-master-$SD_COMMIT-bin-Darwin-macOS-26.6.2-arm64.zip" "$BIN/sdcpp" sd-cli
    fi
    ;;
  Linux)
    kind="$(gpu)"
    note "linux · $ARCH · gpu: $kind (Linux is untested; reports welcome)"
    if [ "$ARCH" = x86_64 ]; then
      case "$kind" in
        cuda) llama="ubuntu-cuda-12.8-x64" ;;
        vulkan) llama="ubuntu-vulkan-x64" ;;
        *) llama="ubuntu-x64" ;;
      esac
      # stable-diffusion.cpp publishes Vulkan and CPU builds for Linux;
      # Vulkan also runs on NVIDIA cards.
      if [ "$kind" = cpu ]; then sd="x86_64"; else sd="x86_64-vulkan"; fi
    else
      case "$kind" in
        cuda) llama="ubuntu-cuda-13.4-arm64" ;;
        vulkan) llama="ubuntu-vulkan-arm64" ;;
        *) llama="ubuntu-arm64" ;;
      esac
      sd=""
    fi
    [ -x "$BIN/llamacpp/llama-server" ] || fetch "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/llama-$LLAMA_TAG-bin-$llama.tar.gz" "$BIN/llamacpp" llama-server
    if [ -n "$sd" ]; then
      [ -x "$BIN/sdcpp/sd-cli" ] || fetch "https://github.com/leejet/stable-diffusion.cpp/releases/download/$SD_TAG/sd-master-$SD_COMMIT-bin-Linux-Ubuntu-24.04-$sd.zip" "$BIN/sdcpp" sd-cli
    else
      note "no prebuilt stable-diffusion.cpp for linux/$ARCH; build it and set SDCPP_BIN for images"
    fi
    ;;
esac

# --------------------------------------------------------------------- app
say "3/5 building Ferro64"
if [ "$OS" = Darwin ]; then
  "$ROOT/scripts/build-ocr.sh" >/dev/null && note "OCR helper: bin/ocr"
fi
(cd "$ROOT" && cargo build --release)

# ------------------------------------------------------------ this machine
say "4/5 what fits this machine (nothing is downloaded)"
"$ROOT/target/release/ferro64" --hardware | sed 's/^/  /'
RAM_GB=0
[ "$OS" = Darwin ] && RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))

# ---------------------------------------------------------------- optional
say "5/5 optional parts"
swift_minor() { swift --version 2>/dev/null | sed -n 's/.*Swift version \([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -1; }
if [ "$WANT_VIDEO" = 1 ]; then
  if [ "$OS" != Darwin ] || [ "$RAM_GB" -lt 30 ]; then
    note "video: skipped. It needs a Mac with 32 GB of memory or more."
  elif ! xcodebuild -version >/dev/null 2>&1; then
    note "video: needs the full Xcode, not only the command line tools."
    note "  1. Install Xcode from the App Store"
    note "  2. sudo xcodebuild -license accept && sudo xcodebuild -runFirstLaunch"
    note "  3. run ./scripts/setup.sh --video again"
  else
    read -r major minor <<< "$(swift_minor)"
    if [ "${major:-0}" -lt 6 ] || { [ "${major:-0}" -eq 6 ] && [ "${minor:-0}" -lt 3 ]; }; then
      note "video: needs Swift 6.3 or newer (this Xcode has ${major:-?}.${minor:-?}). Update Xcode, then run this again."
    else
      "$ROOT/scripts/setup-video.sh"
      note "video engine ready; the model (about 46 GB) downloads with the first video"
    fi
  fi
else
  if [ "$OS" = Darwin ] && [ "$RAM_GB" -ge 30 ]; then
    note "video: run ./scripts/setup.sh --video (needs Xcode)"
  fi
fi
if [ "$WANT_DESKTOP" = 1 ]; then
  if [ "$OS" = Darwin ]; then
    "$ROOT/scripts/build-desktop.sh"
  elif pkg-config --exists webkit2gtk-4.1 2>/dev/null; then
    (cd "$ROOT" && cargo build --release -p ferro64-desktop)
    note "desktop app: target/release/ferro64-desktop"
  else
    note "the desktop app needs WebKitGTK and its build files. Install them, then run this again:"
    note "  Debian/Ubuntu: sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev libsoup-3.0-dev pkg-config build-essential"
    note "  Fedora:        sudo dnf install webkit2gtk4.1-devel gtk3-devel librsvg2-devel libsoup3-devel"
    note "  Arch:          sudo pacman -S webkit2gtk-4.1 gtk3 librsvg libsoup3 base-devel"
  fi
else
  note "desktop app: run ./scripts/setup.sh --desktop"
fi

say "done"
note "start:   cargo run --release   (then open http://localhost:3042)"
note "models:  Settings → Models. The best fits for this machine are marked; ones that won't run are greyed out."
