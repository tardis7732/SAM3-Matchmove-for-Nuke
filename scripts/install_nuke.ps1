[CmdletBinding()]
param(
    [string]$NukeDir = (Join-Path $env:USERPROFILE '.nuke'),
    [string]$PythonExe = '',
    [string]$Checkpoint = ''
)
$ErrorActionPreference = 'Stop'
$pluginRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pluginPath = Join-Path $pluginRoot 'nuke'
if (-not (Test-Path -LiteralPath (Join-Path $pluginPath 'sam3_matchmove_nuke.py') -PathType Leaf)) {
    throw 'Plugin controller is missing.'
}
if ($PythonExe -and -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Worker Python does not exist: $PythonExe"
}
if ($Checkpoint -and -not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "Checkpoint does not exist: $Checkpoint"
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$resolvedNukeDir = [IO.Path]::GetFullPath($NukeDir)
[IO.Directory]::CreateDirectory($resolvedNukeDir) | Out-Null
$startup = Join-Path $resolvedNukeDir 'init.py'
$startMarker = '# >>> SAM3_MATCHMOVE_NUKE >>>'
$endMarker = '# <<< SAM3_MATCHMOVE_NUKE <<<'
$oldText = if (Test-Path -LiteralPath $startup) { [IO.File]::ReadAllText($startup, $utf8) } else { '' }
if ($oldText.Contains($startMarker)) {
    Write-Output "Already registered: $startup"
} else {
    if (Test-Path -LiteralPath $startup) {
        $backup = $startup + '.sam3-backup-' + (Get-Date -Format 'yyyyMMdd_HHmmss_ffff')
        Copy-Item -LiteralPath $startup -Destination $backup
        Write-Output "Backup: $backup"
    }
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pluginPath))
    $block = @"

$startMarker
import base64 as _sam3_b64
import nuke as _sam3_host
_sam3_host.pluginAddPath(_sam3_b64.b64decode('$encoded').decode('utf-8'))
del _sam3_b64, _sam3_host
$endMarker
"@
    # Keep all original startup bytes, including any UTF-8 BOM, and append ASCII only.
    [IO.File]::AppendAllText($startup, "`r`n" + $block + "`r`n", $utf8)
    Write-Output "Registered: $startup"
}
$configPath = Join-Path $pluginRoot 'config.local.json'
if (-not (Test-Path -LiteralPath $configPath)) {
    $cfg = [ordered]@{python_exe=$PythonExe; checkpoint=$Checkpoint; output_root=(Join-Path $pluginRoot 'outputs')}
    [IO.File]::WriteAllText($configPath, ($cfg | ConvertTo-Json), $utf8)
    Write-Output "Configuration: $configPath"
} else {
    Write-Output "Existing configuration retained: $configPath"
}
Write-Output 'Restart Nuke. Menu: Nodes > AI > SAM3 Matchmove.'
