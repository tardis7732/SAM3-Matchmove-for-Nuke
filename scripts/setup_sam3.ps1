<#
.SYNOPSIS
Installs the local Windows SAM3 worker and connects config.local.json to it.
.EXAMPLE
.\scripts\setup_sam3.ps1 -PythonExe 'C:\Python312\python.exe'
.NOTES
Run from PowerShell. PythonExe must be a real Python 3.12+ executable when
creating the environment. Existing environments can be updated without it.
Model access and download are handled separately by auth_sam3.ps1.
#>
[CmdletBinding()]
param([string]$PythonExe = '')

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$pluginRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$venvPath = Join-Path $pluginRoot '.venv-sam3'
$workerPython = Join-Path $venvPath 'Scripts\python.exe'
$vendorPath = Join-Path $pluginRoot 'vendor\sam3'
$pinnedCommit = '660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7'
$officialRepository = 'https://github.com/facebookresearch/sam3.git'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Invoke-NativeChecked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable"
    }
}

function Invoke-Sam3Git {
    param([string[]]$Arguments)
    # Scope ownership trust to this explicit vendor directory and this command.
    # This does not change the user's global Git configuration.
    Invoke-NativeChecked $gitPath (@('-c', "safe.directory=$vendorPath", '-C', $vendorPath) + $Arguments)
}

function Assert-WorkerPython {
    param([string]$Executable)
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "Python executable does not exist: $Executable"
    }
    Invoke-NativeChecked $Executable @(
        '-c',
        'import sys; assert sys.version_info >= (3, 12), "SAM3 requires Python 3.12+"; print("Python:", sys.executable, sys.version.split()[0])'
    )
}

Push-Location -LiteralPath $pluginRoot
try {
    if (-not (Test-Path -LiteralPath $workerPython -PathType Leaf)) {
        if ([string]::IsNullOrWhiteSpace($PythonExe)) {
            throw 'Pass -PythonExe with the full path to a real Python 3.12+ python.exe to create .venv-sam3.'
        }
        $basePython = [IO.Path]::GetFullPath($PythonExe)
        Assert-WorkerPython $basePython
        if (Test-Path -LiteralPath $venvPath) {
            throw "An incomplete environment exists at $venvPath. It was preserved; repair or rename it before running setup again."
        }
        Invoke-NativeChecked $basePython @('-m', 'venv', $venvPath)
    }
    Assert-WorkerPython $workerPython

    $gitPath = (Get-Command git -CommandType Application -ErrorAction Stop).Source
    if (-not (Test-Path -LiteralPath (Join-Path $vendorPath '.git'))) {
        if ((Test-Path -LiteralPath $vendorPath) -and @(Get-ChildItem -LiteralPath $vendorPath -Force).Count -gt 0) {
            throw "The vendor directory is not a Git checkout and is not empty: $vendorPath. Its contents were preserved."
        }
        [IO.Directory]::CreateDirectory((Split-Path -Parent $vendorPath)) | Out-Null
        Invoke-NativeChecked $gitPath @('clone', '--depth', '1', $officialRepository, $vendorPath)
    }

    $origin = (Invoke-Sam3Git @('remote', 'get-url', 'origin') | Out-String).Trim()
    if ($origin.TrimEnd('/') -notin @($officialRepository, 'https://github.com/facebookresearch/sam3')) {
        throw "The SAM3 vendor origin is not the expected official repository: $origin"
    }
    $changes = (Invoke-Sam3Git @('status', '--porcelain', '--untracked-files=no') | Out-String).Trim()
    $currentCommit = (Invoke-Sam3Git @('rev-parse', 'HEAD') | Out-String).Trim()
    if ($changes) {
        throw "The SAM3 vendor checkout has tracked changes. They were preserved: $vendorPath"
    }
    $hasSource = Test-Path -LiteralPath (Join-Path $vendorPath 'sam3\model_builder.py') -PathType Leaf
    if (($currentCommit -ne $pinnedCommit) -or -not $hasSource) {
        Invoke-Sam3Git @('fetch', '--depth', '1', 'origin', $pinnedCommit)
        Invoke-Sam3Git @('checkout', '--detach', $pinnedCommit)
    }
    $installedCommit = (Invoke-Sam3Git @('rev-parse', 'HEAD') | Out-String).Trim()
    if ($installedCommit -ne $pinnedCommit) {
        throw "Unexpected SAM3 revision after checkout: $installedCommit"
    }

    Invoke-NativeChecked $workerPython @('-m', 'pip', 'install', '--no-cache-dir', '--upgrade', 'pip', 'setuptools<81', 'wheel')
    Invoke-NativeChecked $workerPython @(
        '-m', 'pip', 'install', '--no-cache-dir',
        'torch==2.10.0', 'torchvision==0.25.0',
        '--index-url', 'https://download.pytorch.org/whl/cu128'
    )
    Invoke-NativeChecked $workerPython @('-m', 'pip', 'install', '--no-cache-dir', '-r', (Join-Path $pluginRoot 'requirements-sam3-windows.txt'))
    Invoke-NativeChecked $workerPython @('-m', 'pip', 'check')
    Invoke-NativeChecked $workerPython @('-m', 'sam3_matchmove.worker', '--check-sam3')

    $configPath = Join-Path $pluginRoot 'config.local.json'
    $configExists = Test-Path -LiteralPath $configPath -PathType Leaf
    $config = if ($configExists) {
        Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } else {
        [pscustomobject]@{}
    }
    if ($null -eq $config -or $config -isnot [pscustomobject]) {
        throw 'config.local.json must contain a JSON object. The existing file was preserved.'
    }
    # Compare canonical serialization rather than the original file's newline
    # or indentation so a subsequent run does not create another backup.
    $oldJson = $config | ConvertTo-Json -Depth 50
    $newPython = $workerPython.Replace('\', '/')
    $oldPython = if ($config.PSObject.Properties['python_exe']) { [string]$config.python_exe } else { '' }
    if ($oldPython -and $oldPython.Replace('\', '/') -ine $newPython) {
        $previousPythons = @()
        if ($config.PSObject.Properties['runtime_previous_python_exes']) {
            $previousPythons = @($config.runtime_previous_python_exes | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
        }
        if ($previousPythons -notcontains $oldPython) {
            $previousPythons += $oldPython
        }
        $config | Add-Member -NotePropertyName runtime_previous_python_exes -NotePropertyValue $previousPythons -Force
    }
    $config | Add-Member -NotePropertyName python_exe -NotePropertyValue $newPython -Force
    if (-not $config.PSObject.Properties['checkpoint'] -or [string]::IsNullOrWhiteSpace([string]$config.checkpoint)) {
        $checkpointPath = (Join-Path $pluginRoot 'checkpoints\sam3.pt').Replace('\', '/')
        $config | Add-Member -NotePropertyName checkpoint -NotePropertyValue $checkpointPath -Force
    }
    $newJson = $config | ConvertTo-Json -Depth 50
    if (-not $configExists -or $oldJson -cne $newJson) {
        if ($configExists) {
            $backupDirectory = Join-Path $pluginRoot 'backups'
            [IO.Directory]::CreateDirectory($backupDirectory) | Out-Null
            $backupPath = Join-Path $backupDirectory ('config.local.before-sam3-' + (Get-Date -Format 'yyyyMMdd_HHmmss_ffff') + '.json')
            Copy-Item -LiteralPath $configPath -Destination $backupPath
            Write-Output "Configuration backup: $backupPath"
        }
        [IO.File]::WriteAllText($configPath, $newJson + [Environment]::NewLine, $utf8)
    }
    Write-Output "Worker Python: $workerPython"
    Write-Output "SAM3 revision: $installedCommit"
    Write-Output "Configuration: $configPath"
    Write-Output 'Runtime checks passed. This check does not load weights or prove text inference.'
    Write-Output 'Run .\scripts\auth_sam3.ps1 to sign in and download the official checkpoint, then restart Nuke.'
}
finally {
    Pop-Location
}
