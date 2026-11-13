# Windows bootstrap only. The wizard and automated commands share install.py.
param(
    [string]$PythonPath,
    [switch]$InstallPrerequisites,
    [switch]$NonInteractive,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$InstallerArguments
)
$ErrorActionPreference = 'Stop'

function Find-SetupPython {
    if ($PythonPath) {
        if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'The specified Python executable does not exist.' }
        return $PythonPath
    }
    $projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $projectPython) { return $projectPython }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $found = & py -3.11 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $found) { return [string]$found }
        } catch { $found = $null }
    }
    $userPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'
    if (Test-Path -LiteralPath $userPython) { return $userPython }
    return $null
}

try {
    if ($NonInteractive -and -not $InstallerArguments) {
        throw 'Choose an action for automation, for example: -NonInteractive plan --json'
    }
    $python = Find-SetupPython
    $missing = @()
    if (-not $python) { $missing += @('Python.Python.3.11') }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { $missing += @('Git.Git') }
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or
        -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) { $missing += @('Gyan.FFmpeg') }
    $inspection = $InstallerArguments -contains 'plan' -or $InstallerArguments -contains 'check' -or $InstallerArguments -contains '--help'
    if ($missing.Count -and -not $inspection) {
        if (-not $InstallPrerequisites -and -not $NonInteractive) {
            Write-Host "`nSome basic tools are missing: $($missing -join ', ')"
            Write-Host 'Python runs the app, Git downloads updates, and FFmpeg reads video and audio.'
            Write-Host 'Windows Package Manager can install them. Windows may ask for administrator approval.'
            $InstallPrerequisites = (Read-Host 'Install these tools now? Type y to continue').Trim().ToLowerInvariant() -eq 'y'
        }
        if (-not $InstallPrerequisites) { throw 'Missing prerequisites. Use -InstallPrerequisites to allow their installation, or install them yourself.' }
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            throw 'Windows Package Manager is missing. Install Microsoft App Installer from the Microsoft Store, then run this script again.'
        }
        foreach ($package in $missing) {
            & winget install --exact --id $package --source winget --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
            if ($LASTEXITCODE -ne 0) { throw "Could not install $package (exit $LASTEXITCODE). Follow the Windows message, then run setup again." }
        }
        # Refresh only this process, preserving custom tools already on its PATH.
        $env:Path = $env:Path + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
        $python = Find-SetupPython
    }
    if (-not $python) { throw 'Python 3.11 is missing. Run install.ps1 -InstallPrerequisites, or supply -PythonPath to a 64-bit Python 3.11 executable.' }
    $forward = @($InstallerArguments)
    if ($NonInteractive) { $forward += '--non-interactive' }
    & $python -B (Join-Path $PSScriptRoot 'install.py') @forward
    exit $LASTEXITCODE
} catch {
    Write-Host "Setup stopped: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
