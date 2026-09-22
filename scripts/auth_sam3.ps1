<#
.SYNOPSIS
Authenticates interactively to Hugging Face and downloads the official SAM3 checkpoint.
.NOTES
Request model access at https://huggingface.co/facebook/sam3 first.
Enter credentials only in the Hugging Face CLI prompt. This script does not
accept tokens as arguments and does not store tokens in project files.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$pluginRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$hfExe = Join-Path $pluginRoot '.venv-sam3\Scripts\hf.exe'
$checkpointDirectory = Join-Path $pluginRoot 'checkpoints'
$checkpointPath = Join-Path $checkpointDirectory 'sam3.pt'
$expectedSize = [long]3450062241
$modelPage = 'https://huggingface.co/facebook/sam3'

if (-not (Test-Path -LiteralPath $hfExe -PathType Leaf)) {
    throw 'The local Hugging Face CLI is missing. Run scripts\setup_sam3.ps1 first.'
}

try {
    # whoami prints account status, never the token. Authentication is handled
    # by the CLI's interactive prompt and its standard credential store.
    & $hfExe auth whoami
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Model access must be approved for your account: $modelPage"
        & $hfExe auth login
        if ($LASTEXITCODE -ne 0) {
            throw "Hugging Face login failed with exit code $LASTEXITCODE."
        }
    }

    [IO.Directory]::CreateDirectory($checkpointDirectory) | Out-Null
    & $hfExe download facebook/sam3 sam3.pt --local-dir $checkpointDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "The official checkpoint download failed with exit code $LASTEXITCODE."
    }
    if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
        throw "The download did not produce the expected checkpoint: $checkpointPath"
    }
    $actualSize = (Get-Item -LiteralPath $checkpointPath).Length
    if ($actualSize -ne $expectedSize) {
        throw "Unexpected checkpoint size: $actualSize bytes; expected $expectedSize bytes. The file was preserved."
    }
    Write-Output "Official SAM3 checkpoint: $checkpointPath"
    Write-Output "Verified file size: $actualSize bytes. This does not replace a model inference check."
    Write-Output 'Next: .\.venv-sam3\Scripts\python.exe tools/configure.py'
    Write-Output 'Then run the repository-root install.ps1 and restart Nuke. Add SAM3 Mask OFX.'
}
catch {
    Write-Output "SAM3 is gated. Request access with the same Hugging Face account at $modelPage and wait for approval."
    Write-Output 'If access is already approved, check your connection and account permissions, then run this script again.'
    throw
}
