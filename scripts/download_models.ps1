param(
    [string]$SourceDir,
    [string]$UrlPrefix
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$ModelsDir = Join-Path $Root 'models'
$PrimaryReadmeUrl = 'https://github.com/microsoft/DCVC/blob/main/README.md'
$PrimaryShareUrl = 'https://1drv.ms/f/c/2866592d5c55df8c/Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8'
$BackupShareUrl = 'https://1drv.ms/f/c/2866592d5c55df8c/EozfVVwtWWYggCitBAAAAAABbT4z2Z10fMXISnan72UtSA?e=BID7DA'
$RequiredFiles = @(
    'int16_bundle_v1.0.0.pt'
)
$RecommendedFiles = @(
    'cvpr2025_image.pth.tar',
    'cvpr2025_video.pth.tar'
)
$OptionalFiles = @(
    'frozen_entropy.pt'
)

New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

function Test-ModelFiles {
    param([string[]]$Names)
    foreach ($Name in $Names) {
        if (-not (Test-Path (Join-Path $ModelsDir $Name))) {
            return $false
        }
    }
    return $true
}

if (Test-ModelFiles -Names $RequiredFiles) {
    Write-Host "Models already present in: $ModelsDir"
    foreach ($Name in ($RequiredFiles + $RecommendedFiles + $OptionalFiles)) {
        if (Test-Path (Join-Path $ModelsDir $Name)) {
            Write-Host "  found: $Name"
        }
    }
    exit 0
}

if ($SourceDir) {
    $SourceDir = $SourceDir.TrimEnd('\', '/')
    Write-Host "Copying model files from: $SourceDir"
    foreach ($Name in ($RequiredFiles + $RecommendedFiles + $OptionalFiles)) {
        $SourcePath = Join-Path $SourceDir $Name
        if (Test-Path $SourcePath) {
            Copy-Item -Force $SourcePath -Destination (Join-Path $ModelsDir $Name)
            Write-Host "  copied: $Name"
        }
    }
}

if ($UrlPrefix) {
    $UrlPrefix = $UrlPrefix.TrimEnd('/')
    Write-Host "Downloading model files from: $UrlPrefix"
    foreach ($Name in ($RequiredFiles + $RecommendedFiles + $OptionalFiles)) {
        $Uri = "$UrlPrefix/$Name"
        $OutFile = Join-Path $ModelsDir $Name
        Write-Host "  downloading: $Name"
        Invoke-WebRequest -Uri $Uri -OutFile $OutFile
    }
}

if (Test-ModelFiles -Names $RequiredFiles) {
    Write-Host "Model files are ready in: $ModelsDir"
    foreach ($Name in ($RecommendedFiles + $OptionalFiles)) {
        if (Test-Path (Join-Path $ModelsDir $Name)) {
            Write-Host "  found: $Name"
        }
        else {
            Write-Host "  missing (optional): $Name"
        }
    }
    exit 0
}

Write-Host ''
Write-Host 'Automatic download is unreliable for the official OneDrive shares.'
Write-Host ''
Write-Host 'Download the checkpoints from:'
Write-Host "  - $PrimaryReadmeUrl"
Write-Host "  - Primary folder: $PrimaryShareUrl"
Write-Host "  - Backup folder:  $BackupShareUrl"
Write-Host ''
Write-Host 'Place these filenames in:'
Write-Host "  $ModelsDir"
Write-Host ''
Write-Host 'Required:'
Write-Host '  - int16_bundle_v1.0.0.pt'
Write-Host ''
Write-Host 'Recommended:'
Write-Host '  - cvpr2025_image.pth.tar'
Write-Host '  - cvpr2025_video.pth.tar'
Write-Host ''
Write-Host 'Optional:'
Write-Host '  - frozen_entropy.pt'
Write-Host ''
Write-Host 'If you already have the files elsewhere, rerun with:'
Write-Host '  .\scripts\download_models.ps1 -SourceDir C:\path\to\model\folder'
Write-Host ''
Write-Host 'If you have direct file URLs, rerun with:'
Write-Host '  .\scripts\download_models.ps1 -UrlPrefix https://example.com/models'
exit 1
