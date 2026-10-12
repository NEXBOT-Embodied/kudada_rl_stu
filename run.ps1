[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$PythonPath = '',
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)] [string[]]$Arguments
)
$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
if (!$PythonPath) { $PythonPath = Join-Path $repo '.venv\Scripts\python.exe' }
if (!(Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw 'Python not found. Run .\install.ps1, or use .\run.ps1 -PythonPath C:\path\to\python.exe doctor (existing environment).'
}
Push-Location $repo
$previousPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $PythonPath -X utf8 (Join-Path $repo 'course\cli.py') @Arguments 2>&1 | ForEach-Object { Write-Output $_.ToString() }
    $result = $LASTEXITCODE
} finally { $ErrorActionPreference = $previousPreference; Pop-Location }
exit $result
