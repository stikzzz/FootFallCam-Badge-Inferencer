[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$Video,

    [ValidateSet('auto', 'cpu', 'gpu')]
    [string]$DetectorDevice = 'auto',

    [ValidateRange(1, 2147483647)]
    [int]$MaxFrames
)

$ErrorActionPreference = 'Stop'
$videoFile = Get-Item -LiteralPath $Video
if ($videoFile.PSIsContainer) {
    throw "Pass a video file, not a directory: $Video"
}

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Create the project .venv and install requirements-inference.txt before running the demo.'
}
if (-not (Get-Command ffmpeg -CommandType Application -ErrorAction SilentlyContinue)) {
    throw 'FFmpeg must be available on PATH.'
}

$personModel = Join-Path $PSScriptRoot 'output\yolo26s_person_fullframe_training\weights\best_openvino_model'
$badgeModel = Join-Path $PSScriptRoot 'models\yolo26n_badge_stage_openvino_model'
$personFiles = 'best.xml', 'best.bin', 'metadata.yaml'
$badgeFiles = 'yolo26n_badge_stage.xml', 'yolo26n_badge_stage.bin', 'metadata.yaml'
foreach ($file in $personFiles) {
    $path = Join-Path $personModel $file
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing person model file: $path" }
}
foreach ($file in $badgeFiles) {
    $path = Join-Path $badgeModel $file
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing badge model file: $path" }
}

$outputDir = Join-Path $PSScriptRoot 'output\demo'
$name = $videoFile.BaseName + '_badged'
$overlay = Join-Path $outputDir ($name + '.mp4')
$csv = Join-Path $outputDir ($name + '.csv')
if ((Test-Path -LiteralPath $overlay) -or (Test-Path -LiteralPath $csv)) {
    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $overlay = Join-Path $outputDir ($name + '_' + $timestamp + '.mp4')
    $csv = Join-Path $outputDir ($name + '_' + $timestamp + '.csv')
}

$inferenceArgs = @(
    (Join-Path $PSScriptRoot 'infer_staff.py')
    $videoFile.FullName
    '--detector', $personModel
    '--detector-device', $DetectorDevice
    '--imgsz', '640'
    '--conf', '0.10'
    '--badge-detector', $badgeModel
    '--badge-imgsz', '288'
    '--badge-conf', '0.40'
    '--label-badged'
    '--output', $overlay
    '--csv', $csv
)
if ($PSBoundParameters.ContainsKey('MaxFrames')) {
    $inferenceArgs += '--max-frames', $MaxFrames.ToString()
}
& $python @inferenceArgs
if ($LASTEXITCODE -ne 0) {
    throw "Badge inference exited with code $LASTEXITCODE."
}
