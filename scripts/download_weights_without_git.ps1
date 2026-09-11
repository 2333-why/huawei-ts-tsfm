param(
    [string]$WorkDir = (Join-Path (Get-Location) "huawei-ts-tsfm-weights")
)

$ErrorActionPreference = "Stop"

$WorkDir = [System.IO.Path]::GetFullPath($WorkDir)
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
Set-Location $WorkDir

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCommand) {
    throw "Python is required. Activate the environment that should run the download, then retry."
}
$Python = $PythonCommand.Source

& $Python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.9 or newer is required. Activate a compatible environment, then retry."
}

Write-Host "Using active Python: $Python"
& $Python -m pip install --upgrade huggingface_hub
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install huggingface_hub."
}

$env:HF_HOME = Join-Path $WorkDir "checkpoints_huggingface"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "600"
$env:HF_HUB_ETAG_TIMEOUT = "60"

$SecureToken = Read-Host "Hugging Face read token (input hidden)" -AsSecureString
$env:HF_TOKEN = [System.Net.NetworkCredential]::new("", $SecureToken).Password

$DownloadCode = @'
import hashlib
import os
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = (
    (
        "Sundial",
        "thuml/sundial-base-128m",
        "3212e42564493f520593e5414af4367fc4b49226",
    ),
    (
        "TimeMoE",
        "Maple728/TimeMoE-50M",
        "446753ee48ff3726d0606a81d0092d54acee995e",
    ),
    (
        "Chronos2",
        "amazon/chronos-2",
        "29ec3766d36d6f73f0696f85560a422f50e8498c",
    ),
    (
        "TiRex",
        "NX-AI/TiRex",
        "63c740922493f5fbe60b277609ec62babfba2762",
    ),
    (
        "TimesFM",
        "google/timesfm-2.5-200m-transformers",
        "5a9806b9b291fad9233b5249d88263f1846304d3",
    ),
)

work_dir = Path.cwd()
cache_dir = Path(os.environ["HF_HOME"]).resolve()
archive = work_dir / "huawei-ts-tsfm-hf-cache.tar.gz"
checksum = work_dir / "huawei-ts-tsfm-hf-cache.tar.gz.sha256"

cache_dir.mkdir(parents=True, exist_ok=True)

for name, repo_id, revision in MODELS:
    print(f"\nDownloading {name}: {repo_id}@{revision}", flush=True)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir / "hub",
        token=os.environ.get("HF_TOKEN"),
        max_workers=4,
    )
    print(f"Ready {name}: {path}", flush=True)

print(f"\nCreating archive: {archive}", flush=True)
with tarfile.open(
    archive,
    mode="w:gz",
    compresslevel=1,
    dereference=False,
) as bundle:
    bundle.add(cache_dir, arcname="checkpoints_huggingface", recursive=True)

digest = hashlib.sha256()
with archive.open("rb") as stream:
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(block)

checksum.write_text(
    f"{digest.hexdigest()}  {archive.name}\n",
    encoding="ascii",
)

print("\nAll five model checkpoints downloaded.")
print(f"Archive: {archive}")
print(f"SHA256: {checksum}")
'@

try {
    & $Python -c $DownloadCode
    if ($LASTEXITCODE -ne 0) {
        throw "Weight download or archive creation failed. Run this script again to resume."
    }
}
finally {
    Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
    $SecureToken = $null
}

Get-Item -LiteralPath `
    (Join-Path $WorkDir "huawei-ts-tsfm-hf-cache.tar.gz"), `
    (Join-Path $WorkDir "huawei-ts-tsfm-hf-cache.tar.gz.sha256") |
    Select-Object FullName, Length
