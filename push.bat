@echo off
title Push MedSeg-XAI to GitHub
echo =========================================================
echo   Pushing MedSeg-XAI to https://github.com/ash333xm/medseg-xai
echo =========================================================
echo.
cd /d "D:\ayush_medseg_workflow"

git add .
git commit -m "feat(clinical-data): replace mock data with authentic Google Drive clinical cases, outputs, heatmaps, segmentations, and audit records"
echo.
echo Pushing to origin main...
git push origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =========================================================
    echo   SUCCESS! Pushed to https://github.com/ash333xm/medseg-xai
    echo   Vercel will now automatically deploy all clinical datasets!
    echo =========================================================
) else (
    echo.
    echo Pushing failed or requires authentication.
    echo If prompted by browser or Git Credential Manager, please log in.
)

echo.
pause
