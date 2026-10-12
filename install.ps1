[CmdletBinding()]
param(
    # Used only to create .venv; dependencies always go into .venv.
    [string]$PythonPath = '',
    [string]$PackageIndex = 'https://mirrors.ustc.edu.cn/pypi/simple',
    [string]$TorchIndex = 'https://mirrors.nju.edu.cn/pytorch/whl/cpu'
)
$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$venv = Join-Path $repo '.venv'
$targetPython = Join-Path $venv 'Scripts\python.exe'

if (!(Test-Path -LiteralPath $venv)) {
    if ($PythonPath) {
        $PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
        & $PythonPath -c 'import sys; assert sys.version_info[:2] == (3, 10) and sys.maxsize > 2**32, sys.version'
        if ($LASTEXITCODE) { throw '64-bit Python 3.10.x is required.' }
        & $PythonPath -m venv $venv
    } else {
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if (!$uv) { throw 'Install uv, or specify -PythonPath pointing to 64-bit Python 3.10.x to create .venv.' }
        $previousPythonMirror = $env:UV_PYTHON_INSTALL_MIRROR
        try {
            if (!$env:UV_PYTHON_INSTALL_MIRROR) {
                $env:UV_PYTHON_INSTALL_MIRROR = 'https://mirrors.nju.edu.cn/github-release/astral-sh/python-build-standalone'
            }
            & $uv.Source venv --seed --python 3.10.21 --index-url $PackageIndex $venv
        } finally { $env:UV_PYTHON_INSTALL_MIRROR = $previousPythonMirror }
    }
    if ($LASTEXITCODE) { throw 'Cannot create .venv.' }
} else {
    Write-Host "Reusing: $venv"
}
if (!(Test-Path -LiteralPath $targetPython -PathType Leaf)) {
    throw 'Incomplete .venv. Rename it to keep a backup, then rerun install.ps1.'
}

# Windows PowerShell 5.1 can treat redirected native stderr as errors.
# Preserve the actual Python process exit code, including pip/doctor failures.
$previousPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $targetPython -X utf8 (Join-Path $repo 'scripts\environment.py') install --package-index $PackageIndex --torch-index $TorchIndex 2>&1 |
        ForEach-Object { Write-Output $_.ToString() }
    $result = $LASTEXITCODE
} finally { $ErrorActionPreference = $previousPreference }
if ($result) { throw 'Environment installation failed. For a broken .venv, rename it and rerun install.ps1.' }
Write-Host 'Ready. Run: .\run.ps1 doctor'
