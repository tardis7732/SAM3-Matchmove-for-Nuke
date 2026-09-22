param(
    [switch]$BuildFromSource,
    [switch]$SkipBuild,
    [string]$BuildDirectory = '',
    [string]$NukeUserDir = (Join-Path $env:USERPROFILE '.nuke')
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$profile = Join-Path $projectRoot 'config/frontend.json'
if (-not (Test-Path -LiteralPath $profile -PathType Leaf)) {
    throw 'Run tools/configure.py with your external SAM3 Python before installing OFX. See README.md.'
}
$settings = Get-Content -LiteralPath $profile -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $BuildDirectory) { $BuildDirectory = if ($BuildFromSource) { 'build' } else { 'prebuilt' } }
if ($BuildFromSource -and -not $SkipBuild) {
    & (Join-Path $projectRoot 'ofx/build.ps1') -BuildDirectory $BuildDirectory
}
$buildRoot = Join-Path (Join-Path $projectRoot 'ofx') $BuildDirectory
$bundle = Join-Path $buildRoot 'SAM3Mask.ofx.bundle/Contents/Win64'
if (-not (Test-Path -LiteralPath (Join-Path $bundle 'SAM3Mask.ofx') -PathType Leaf)) {
    throw 'SAM3Mask.ofx is missing. Run install.ps1 -BuildFromSource with Visual Studio C++ Build Tools installed.'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$configText = "root=$($projectRoot.Replace('\','/'))`npython=$($settings.python.Replace('\','/'))`ndaemon=$($projectRoot.Replace('\','/'))/daemon/launcher.py`nport=47823`n"
[IO.File]::WriteAllText((Join-Path $bundle 'sam3mask.cfg'), $configText, $utf8)
$settings.ofx_build_dir = $buildRoot
[IO.File]::WriteAllText($profile, ($settings | ConvertTo-Json), $utf8)
[IO.Directory]::CreateDirectory($NukeUserDir) | Out-Null
$init = Join-Path $NukeUserDir 'init.py'
$pathLiteral = ConvertTo-Json -InputObject ((Join-Path $projectRoot 'nuke').Replace('\','/')) -Compress
$line = "nuke.pluginAddPath($pathLiteral)"
$existing = if (Test-Path -LiteralPath $init) { [IO.File]::ReadAllText($init) } else { '' }
if (-not $existing.Contains($line)) {
    if (Test-Path -LiteralPath $init) {
        Copy-Item -LiteralPath $init -Destination "$init.sam3-ofx-backup-$(Get-Date -Format yyyyMMddHHmmss)"
    }
    [IO.File]::WriteAllText($init, $existing + "`n# SAM3 Mask OFX`nimport nuke`n$line`n", $utf8)
}
Write-Host 'SAM3 Mask OFX installed. Restart Nuke, then Tab > SAM3 Mask OFX.'
