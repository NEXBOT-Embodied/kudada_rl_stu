param(
    [string]$CondaRoot = (Join-Path $env:USERPROFILE 'miniconda3'),
    [string]$EnvironmentName = 'kudada_course',
    [string]$PythonPath = '',
    [string]$PackageIndex = 'https://pypi.org/simple'
)
$ErrorActionPreference = 'Stop'
if (!$PythonPath) {
    $conda = Join-Path $CondaRoot 'Scripts\conda.exe'
    if (!(Test-Path -LiteralPath $conda)) { throw 'Conda executable not found; specify -CondaRoot or -PythonPath.' }
    $environments = (& $conda env list --json | ConvertFrom-Json).envs
    if ($LASTEXITCODE) { throw 'Cannot list Conda environments' }
    $prefix = $environments | Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName } | Select-Object -First 1
    if (!$prefix) {
        & $conda create --name $EnvironmentName python=3.10.21 pip -y
        if ($LASTEXITCODE) { throw 'Cannot create Conda environment' }
        $environments = (& $conda env list --json | ConvertFrom-Json).envs
        $prefix = $environments | Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName } | Select-Object -First 1
    }
    $PythonPath = Join-Path $prefix 'python.exe'
}
& (Join-Path $PSScriptRoot 'install.ps1') -PythonPath $PythonPath -ExistingEnvironment -PackageIndex $PackageIndex
if ($LASTEXITCODE) { throw 'Conda dependency installation failed' }
