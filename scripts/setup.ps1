# One-time setup for Ferro64 on Windows, after `git clone`. Safe to run again.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1             Rust, the engines, the app, what fits
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Desktop    also the desktop app
#
# Installs what is missing:
#   Rust (rustup) and the Microsoft C++ build tools it needs, with winget
#   llama-server (llama.cpp)        chat, code, agents   (CUDA, Vulkan or CPU build)
#   sd-cli (stable-diffusion.cpp)   images               (CUDA, Vulkan or CPU build)
# It downloads no models: Settings -> Models shows which ones fit this machine.
# Documents (Apple Vision) and video (MLX) are macOS only.
# Windows support is new and untested on real hardware; reports welcome.
param([switch]$Desktop)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is far faster without it

$LlamaTag = if ($env:LLAMA_TAG) { $env:LLAMA_TAG } else { 'b12032' }
$SdTag = if ($env:SD_TAG) { $env:SD_TAG } else { 'master-1039-290148d' }
$SdCommit = $SdTag.Split('-')[-1]
$Root = Split-Path -Parent $PSScriptRoot
$Bin = Join-Path $Root 'data\bin'
$Arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'x64' }

function Say($text) { Write-Host "`n$text" -ForegroundColor White }
function Note($text) { Write-Host "  $text" }
function Has($command) { [bool](Get-Command $command -ErrorAction SilentlyContinue) }

# Downloads a zip and copies the folder holding $Want into $Dest.
function Fetch($Url, $Dest, $Want) {
  $tmp = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid())
  New-Item -ItemType Directory -Path $tmp | Out-Null
  Note "downloading $(Split-Path -Leaf $Url)"
  Invoke-WebRequest -Uri $Url -OutFile "$tmp\archive.zip" -UseBasicParsing
  Expand-Archive -Path "$tmp\archive.zip" -DestinationPath "$tmp\x" -Force
  $found = Get-ChildItem -Path "$tmp\x" -Recurse -Filter $Want | Select-Object -First 1
  if (-not $found) { throw "$Want not found in $Url" }
  New-Item -ItemType Directory -Path $Dest -Force | Out-Null
  Copy-Item -Path (Join-Path $found.DirectoryName '*') -Destination $Dest -Recurse -Force
  Remove-Item -Recurse -Force $tmp
}

# Extra runtime files (CUDA DLLs) go next to the engine.
function FetchInto($Url, $Dest) {
  $tmp = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid())
  New-Item -ItemType Directory -Path $tmp | Out-Null
  Note "downloading $(Split-Path -Leaf $Url)"
  Invoke-WebRequest -Uri $Url -OutFile "$tmp\archive.zip" -UseBasicParsing
  Expand-Archive -Path "$tmp\archive.zip" -DestinationPath "$tmp\x" -Force
  Get-ChildItem -Path "$tmp\x" -Recurse -Filter '*.dll' | Copy-Item -Destination $Dest -Force
  Remove-Item -Recurse -Force $tmp
}

function Gpu {
  if (Has 'nvidia-smi') { return 'cuda' }
  $names = (Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) -join ' '
  if ($names -match 'AMD|Radeon|Intel\(R\) Arc|Arc\(TM\)|Iris') { return 'vulkan' }
  return 'cpu'
}

# ---------------------------------------------------------------- toolchain
Say '1/5 toolchain'
if (-not (Has 'winget') -and -not (Has 'cargo')) {
  throw 'winget is needed to install Rust. Install "App Installer" from the Microsoft Store, or Rust from https://rustup.rs, then run this again.'
}
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vcTools = if (Test-Path $vswhere) { & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath } else { $null }
if (-not $vcTools) {
  Note 'installing the Microsoft C++ build tools (Rust needs them; a few GB, several minutes)'
  winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --accept-package-agreements --accept-source-agreements `
    --override '--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
}
$cargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
if (-not (Has 'cargo') -and (Test-Path (Join-Path $cargoBin 'cargo.exe'))) { $env:Path = "$cargoBin;$env:Path" }
if (-not (Has 'cargo')) {
  Note 'installing Rust with rustup'
  winget install --id Rustlang.Rustup --exact --accept-package-agreements --accept-source-agreements
  $env:Path = "$cargoBin;$env:Path"
  rustup default stable
}
Note (cargo --version)

# ----------------------------------------------------------------- engines
Say '2/5 engines'
$kind = Gpu
Note "windows · $Arch · gpu: $kind"
$llamaDir = Join-Path $Bin 'llamacpp'
if (Test-Path (Join-Path $llamaDir 'llama-server.exe')) {
  Note 'llama.cpp: data\bin\llamacpp'
} else {
  $base = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag"
  if ($Arch -eq 'arm64') {
    $flavor = if ($kind -eq 'cpu') { 'cpu-arm64' } else { 'vulkan-arm64' }
    Fetch "$base/llama-$LlamaTag-bin-win-$flavor.zip" $llamaDir 'llama-server.exe'
  } elseif ($kind -eq 'cuda') {
    Fetch "$base/llama-$LlamaTag-bin-win-cuda-12.4-x64.zip" $llamaDir 'llama-server.exe'
    FetchInto "$base/cudart-llama-bin-win-cuda-12.4-x64.zip" $llamaDir
  } else {
    Fetch "$base/llama-$LlamaTag-bin-win-$kind-x64.zip" $llamaDir 'llama-server.exe'
  }
}
$sdDir = Join-Path $Bin 'sdcpp'
if (Test-Path (Join-Path $sdDir 'sd-cli.exe')) {
  Note 'stable-diffusion.cpp: data\bin\sdcpp'
} elseif ($Arch -eq 'arm64') {
  Note 'no prebuilt stable-diffusion.cpp for Windows on ARM; images are off'
} else {
  $base = "https://github.com/leejet/stable-diffusion.cpp/releases/download/$SdTag"
  if ($kind -eq 'cuda') {
    Fetch "$base/sd-master-$SdCommit-bin-win-cuda12-x64.zip" $sdDir 'sd-cli.exe'
    FetchInto "$base/cudart-sd-bin-win-cu12-x64.zip" $sdDir
  } else {
    Fetch "$base/sd-master-$SdCommit-bin-win-$kind-x64.zip" $sdDir 'sd-cli.exe'
  }
}

# --------------------------------------------------------------------- app
Say '3/5 building Ferro64'
Push-Location $Root
try { cargo build --release; if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' } } finally { Pop-Location }

# ------------------------------------------------------------ this machine
Say '4/5 what fits this machine (nothing is downloaded)'
& (Join-Path $Root 'target\release\ferro64.exe') --hardware | ForEach-Object { "  $_" }

# ---------------------------------------------------------------- optional
Say '5/5 optional parts'
if ($Desktop) {
  Push-Location $Root
  try { cargo build --release -p ferro64-desktop; if ($LASTEXITCODE -ne 0) { throw 'desktop build failed' } } finally { Pop-Location }
  Note 'desktop app: target\release\ferro64-desktop.exe (uses the WebView2 runtime built into Windows 10 and 11)'
} else {
  Note 'desktop app: run this script again with -Desktop'
}
Note 'documents and video are macOS only for now'

Say 'done'
Note 'start:   cargo run --release   (then open http://localhost:3042)'
Note 'models:  Settings -> Models. The best fits for this machine are marked; ones that will not run are greyed out.'
