$ErrorActionPreference = "Stop"

Write-Host "Enabling Windows Subsystem for Linux..."
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
if ($LASTEXITCODE -notin 0, 3010) {
    throw "Failed to enable Microsoft-Windows-Subsystem-Linux (exit $LASTEXITCODE)"
}

Write-Host "Enabling Virtual Machine Platform..."
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
if ($LASTEXITCODE -notin 0, 3010) {
    throw "Failed to enable VirtualMachinePlatform (exit $LASTEXITCODE)"
}

Write-Host "WSL Windows features enabled. Restart Windows before continuing."
