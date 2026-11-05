#!/usr/bin/env bash
# Builds the video engine: ltx-video (LTX-2.3 on Apple Silicon with MLX, in
# Swift) from a pinned release into data/bin/ltx. Needs Xcode with Swift 6.3
# or newer and its Metal toolchain. The model (~46 GB) downloads into
# data/models/video on the first video.
set -euo pipefail
TAG="${LTX_TAG:-v0.4.2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

xcrun metal --version >/dev/null 2>&1 || xcodebuild -downloadComponent MetalToolchain
git clone --quiet --depth 1 --branch "$TAG" https://github.com/VincentGourbin/ltx-video-swift-mlx.git "$WORK/src"
cd "$WORK/src"
# The two -skip flags approve mlx-swift's build plugin and mlx-swift-lm's
# macros, which a non-interactive build cannot prompt for.
xcodebuild -scheme ltx-video -configuration Release -derivedDataPath .xcodebuild \
  -destination 'platform=macOS' -skipPackagePluginValidation -skipMacroValidation build -quiet

mkdir -p "$ROOT/data/bin/ltx" "$ROOT/data/models/video"
cp .xcodebuild/Build/Products/Release/ltx-video "$ROOT/data/bin/ltx/"
# The Metal shader library must sit next to the binary.
cp -R .xcodebuild/Build/Products/Release/*.bundle "$ROOT/data/bin/ltx/"
echo "installed $ROOT/data/bin/ltx/ltx-video ($TAG)"
