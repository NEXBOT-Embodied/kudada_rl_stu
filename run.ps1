[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)] [string[]]$Arguments
)
$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$PythonPath = Join-Path $repo '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw 'Repository .venv Python not found. Run .\install.ps1 first.'
}
Push-Location $repo
$previousPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $PythonPath -X utf8 (Join-Path $repo 'scripts\environment.py') check 2>&1 | ForEach-Object { Write-Output $_.ToString() }
    if ($LASTEXITCODE) { throw 'Invalid .venv. Rename it to keep a backup, then rerun install.ps1.' }
    & $PythonPath -X utf8 (Join-Path $repo 'course\cli.py') @Arguments 2>&1 | ForEach-Object { Write-Output $_.ToString() }
    $result = $LASTEXITCODE
} finally { $ErrorActionPreference = $previousPreference; Pop-Location }
exit $result
