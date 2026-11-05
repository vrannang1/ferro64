#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p bin
task_module_cache="${TMPDIR:-/tmp}/ferro64-swift-module-cache"
mkdir -p "$task_module_cache"
swiftc -module-cache-path "$task_module_cache" -O -framework AppKit -framework Vision -framework PDFKit tools/ocr.swift -o bin/ocr
printf 'Built bin/ocr\n'
