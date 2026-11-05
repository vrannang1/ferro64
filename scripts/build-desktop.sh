#!/usr/bin/env bash
# Builds dist/Ferro64.app: the desktop window plus the Ferro64 server.
# The server keeps using this folder's data/, models and settings.
# Move the app to /Applications or the Dock if you like; rebuild after
# updating the code.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/Ferro64.app"
cd "$ROOT"
cargo build --release
cargo build --release -p ferro64-desktop

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/release/ferro64-desktop "$APP/Contents/MacOS/Ferro64"
cp target/release/ferro64 "$APP/Contents/MacOS/ferro64"

ICONSET="$(mktemp -d)/icon.iconset"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
  sips -z $size $size desktop/icons/icon.png --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  sips -z $((size * 2)) $((size * 2)) desktop/icons/icon.png --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/icon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Ferro64</string>
  <key>CFBundleDisplayName</key><string>Ferro64</string>
  <key>CFBundleIdentifier</key><string>app.ferro64.desktop</string>
  <key>CFBundleExecutable</key><string>Ferro64</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

# Ad-hoc signature so macOS runs a locally built app.
codesign --force --deep --sign - "$APP" >/dev/null
echo "built $APP"
