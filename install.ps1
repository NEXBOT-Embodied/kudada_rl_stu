param(
    [string]$PythonPath = '',
    [string]$PackageIndex = 'https://pypi.org/simple',
    [switch]$ExistingEnvironment,
    [switch]$Cuda
)
$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
if (!$PythonPath) {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (!$uv) { throw 'Install uv, or specify -PythonPath pointing to Python 3.10.x.' }
    if ($ExistingEnvironment) { throw '-ExistingEnvironment requires -PythonPath.' }
    & $uv.Source venv --seed --python 3.10.21 (Join-Path $repo '.venv')
    if ($LASTEXITCODE) { throw 'Python 3.10 environment creation failed' }
    $targetPython = Join-Path $repo '.venv\Scripts\python.exe'
} else {
    $PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
    & $PythonPath -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
    if ($LASTEXITCODE) { throw 'Python 3.10.x is required.' }
    if ($ExistingEnvironment) { $targetPython = $PythonPath }
    else {
        $targetPython = Join-Path $repo '.venv\Scripts\python.exe'
        if (!(Test-Path -LiteralPath $targetPython)) {
            & $PythonPath -m venv (Join-Path $repo '.venv')
            if ($LASTEXITCODE) { throw 'Virtual environment creation failed' }
        }
    }
}
# Install the course's two small vendored Motrix source packages. No clone or
# editable path to a neighboring research workspace is needed.
function Invoke-CoursePython {
    param([string[]]$PythonArguments)
    $previousPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 can turn redirected native stderr warnings
        # into terminating errors. The process exit code is authoritative.
        $ErrorActionPreference = 'Continue'
        & $targetPython @PythonArguments 2>&1 | ForEach-Object { Write-Output $_.ToString() }
        $script:CoursePythonExit = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
}
$dependencyList = Join-Path $repo 'requirements.lock.txt'
if (!(Test-Path -LiteralPath $dependencyList)) { $dependencyList = Join-Path $repo 'requirements.txt' }
Invoke-CoursePython -PythonArguments @('-m', 'pip', 'install', '-r', $dependencyList, '--index-url', $PackageIndex)
if ($script:CoursePythonExit) { throw 'Course dependencies installation failed' }
Invoke-CoursePython -PythonArguments @('-c', "import importlib.util,sys; spec=importlib.util.find_spec('torch'); sys.exit(2) if spec is None else None; import torch; sys.exit(0 if torch.__version__.split('+')[0] == '2.7.0' else 2)")
if ($script:CoursePythonExit) {
    $torchIndex = if ($Cuda) { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
    Invoke-CoursePython -PythonArguments @('-m', 'pip', 'install', 'torch==2.7.0', '--index-url', $torchIndex)
    if ($script:CoursePythonExit) { throw 'PyTorch installation failed' }
}
Invoke-CoursePython -PythonArguments @('-m', 'pip', 'install', '--no-deps', (Join-Path $repo 'vendor\motrix_env_core'), (Join-Path $repo 'vendor\motrix_env_motrixsim'), '--index-url', $PackageIndex)
if ($script:CoursePythonExit) { throw 'Vendored Motrix source installation failed' }
Invoke-CoursePython -PythonArguments @('-m', 'pip', 'check')
if ($script:CoursePythonExit) { throw 'Dependency consistency check failed' }
Invoke-CoursePython -PythonArguments @('-X', 'utf8', (Join-Path $repo 'course\cli.py'), 'doctor')
if ($script:CoursePythonExit) { throw 'Environment doctor failed' }
Write-Host "Ready: $targetPython"
if ($ExistingEnvironment) { Write-Host "Use: .\run.ps1 -PythonPath '$targetPython' doctor" }
