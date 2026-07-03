@echo off
setlocal

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" -m PyInstaller --noconfirm CouplerGuardAI.spec

if not exist "dist\CouplerGuardAI\models" mkdir "dist\CouplerGuardAI\models"
if not exist "dist\CouplerGuardAI\outputs" mkdir "dist\CouplerGuardAI\outputs"
if not exist "dist\CouplerGuardAI\raw_videos" mkdir "dist\CouplerGuardAI\raw_videos"
if not exist "dist\CouplerGuardAI\raw_photos" mkdir "dist\CouplerGuardAI\raw_photos"
if not exist "dist\CouplerGuardAI\reference_images" mkdir "dist\CouplerGuardAI\reference_images"

copy /Y "camera_config.json" "dist\CouplerGuardAI\camera_config.json" >nul
if exist "models\best.pt" copy /Y "models\best.pt" "dist\CouplerGuardAI\models\best.pt" >nul
if exist "models\classifier.pt" copy /Y "models\classifier.pt" "dist\CouplerGuardAI\models\classifier.pt" >nul
if exist "reference_images\desired_condition.jpg" copy /Y "reference_images\desired_condition.jpg" "dist\CouplerGuardAI\reference_images\desired_condition.jpg" >nul

echo.
echo Build complete. Check the dist\CouplerGuardAI folder.
echo Edit dist\CouplerGuardAI\camera_config.json with the Ethernet camera stream URL before running on the standalone device.
