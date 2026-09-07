# PowerShell helper script to push MedSeg-XAI to GitHub
param (
    [Parameter(Position=0)]
    [string]$RemoteUrl,
    
    [Parameter(Position=1)]
    [string]$Token
)

# Ensure Git is in PATH
$gitCmd = "C:\Users\pujas\AppData\Local\Programs\Git\cmd"
if ($env:Path -notlike "*$gitCmd*") {
    $env:Path = "$gitCmd;" + $env:Path
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       MedSeg-XAI: Push Local Repository to GitHub        " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git executable not found. Please verify MinGit installation."
    exit 1
}

# 1. Prompt for Remote URL if not provided
if (-not $RemoteUrl) {
    Write-Host "Please enter your GitHub repository URL:" -ForegroundColor Yellow
    Write-Host "Example: https://github.com/your-username/medseg-xai.git" -ForegroundColor Gray
    $RemoteUrl = Read-Host "GitHub Remote URL"
}

if (-not $RemoteUrl) {
    Write-Error "No GitHub repository URL provided. Aborting."
    exit 1
}

# 2. Handle GitHub Personal Access Token if provided for HTTPS
$TargetUrl = $RemoteUrl
if ($Token -and $RemoteUrl.StartsWith("https://github.com/")) {
    $TargetUrl = $RemoteUrl.Replace("https://github.com/", "https://$Token@github.com/")
}

# 3. Configure Remote Origin
Write-Host "`n[1/3] Configuring remote 'origin'..." -ForegroundColor Green
git remote remove origin 2>$null
git remote add origin $TargetUrl
Write-Host "Origin set to: $RemoteUrl"

# 4. Push Branches
Write-Host "`n[2/3] Pushing 'main' branch..." -ForegroundColor Green
git push -u origin main

Write-Host "`n[3/3] Pushing 'develop' and 'feature/m1-model-arch' branches..." -ForegroundColor Green
git push -u origin develop
git push -u origin feature/m1-model-arch

Write-Host "`n[SUCCESS] All branches pushed to GitHub successfully!" -ForegroundColor Green
Write-Host "View your repository at: $($RemoteUrl -replace '\.git$', '')" -ForegroundColor Cyan
