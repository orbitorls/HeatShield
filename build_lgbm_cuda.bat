@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Building LightGBM with CUDA (USE_CUDA=1)
echo ============================================================

REM 1. Set up MSVC x64 environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %errorlevel% neq 0 (
    echo ERROR: vcvars64.bat failed. Install "Desktop development with C++" in VS Build Tools.
    pause & exit /b 1
)

REM ---------------------------------------------------------------------------
REM Configurable paths — override via environment variables before running:
REM   set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
REM   set NINJA=%USERPROFILE%\bin\ninja.exe
REM   set LGBM_SRC=C:\src\LightGBM
REM ---------------------------------------------------------------------------
if "%CUDA_ROOT%"=="" (
    REM Default: look for any installed CUDA version under the standard location.
    REM Prefer the newest one by sorting in reverse and taking the first result.
    for /f "delims=" %%d in ('dir /b /ad /o-n "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" 2^>nul') do (
        if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%d\bin\nvcc.exe" (
            set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%d
            goto :cuda_found
        )
    )
    echo ERROR: No CUDA installation found under "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\".
    echo        Install CUDA Toolkit or set CUDA_ROOT before running this script.
    pause & exit /b 1
    :cuda_found
    echo [INFO] Auto-detected CUDA at: %CUDA_ROOT%
)

if "%NINJA%"=="" (
    REM Default: look for ninja.exe next to this script, then on PATH.
    set NINJA=%~dp0ninja.exe
    if not exist "!NINJA!" (
        where ninja.exe >nul 2>&1
        if %errorlevel% equ 0 (
            for /f "delims=" %%p in ('where ninja.exe') do set NINJA=%%p
        ) else (
            echo ERROR: ninja.exe not found.
            echo        Place ninja.exe in the same directory as this script,
            echo        add it to PATH, or set NINJA=^<path^> before running.
            pause & exit /b 1
        )
    )
    echo [INFO] Using ninja at: !NINJA!
)

if "%LGBM_SRC%"=="" (
    REM Default: LightGBM subfolder next to this script.
    set LGBM_SRC=%~dp0LightGBM
    echo [INFO] Using LGBM_SRC: !LGBM_SRC!
)

set LGBM_BUILD=%LGBM_SRC%\build

REM 2. Verify required files exist
if not exist "%CUDA_ROOT%\bin\nvcc.exe" (
    echo ERROR: nvcc.exe not found at %CUDA_ROOT%\bin\nvcc.exe
    echo        Check your CUDA installation or update CUDA_ROOT.
    pause & exit /b 1
)
if not exist "%NINJA%" (
    echo ERROR: ninja.exe not found at %NINJA%
    pause & exit /b 1
)
if not exist "%LGBM_SRC%\CMakeLists.txt" (
    echo ERROR: LightGBM source not found at %LGBM_SRC%
    echo        Clone it with: git clone --recursive https://github.com/microsoft/LightGBM.git
    pause & exit /b 1
)
echo [OK] Tools found: cl.exe, nvcc.exe, ninja.exe

REM 3. Clean build dir
if exist "%LGBM_BUILD%" rmdir /s /q "%LGBM_BUILD%"
mkdir "%LGBM_BUILD%"
cd /d "%LGBM_BUILD%"

REM 4. CMake configure
echo.
echo [CMake] Configuring with CUDA support...
cmake "%LGBM_SRC%" ^
    -G "Ninja" ^
    -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
    -DUSE_CUDA=1 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_ROOT%" ^
    -DCMAKE_CUDA_COMPILER="%CUDA_ROOT%\bin\nvcc.exe"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: CMake configure failed! See error above.
    pause & exit /b 1
)
echo [OK] CMake configure succeeded.

REM 5. Build
echo.
echo [Build] Compiling... (5-15 min, please wait)
"%NINJA%" -j4
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed! See error above.
    pause & exit /b 1
)
echo [OK] Build succeeded!

REM 6. Copy DLL + pip install
echo.
echo [Install] Installing Python package...
copy /y "%LGBM_BUILD%\lib_lightgbm.dll" "%LGBM_SRC%\python-package\lightgbm\lib_lightgbm.dll"
python -m pip install "%LGBM_SRC%\python-package" --force-reinstall --no-deps
if %errorlevel% neq 0 (
    echo ERROR: pip install failed
    pause & exit /b 1
)
echo [OK] Python package installed.

REM 7. Quick test
echo.
echo [Test] Verifying CUDA in LightGBM...
python -c "import numpy as np,lightgbm as lgb; X=np.random.rand(2000,30).astype('f'); y=np.random.rand(2000).astype('f'); d=lgb.Dataset(X,y); lgb.train({'device_type':'cuda','verbose':-1,'num_leaves':32},d,num_boost_round=20); print('[OK] LightGBM CUDA works!')"
if %errorlevel% neq 0 (
    echo ERROR: CUDA test failed
    pause & exit /b 1
)

echo.
echo ============================================================
echo  DONE! Now run training:
echo    python scripts/auto_tune_models.py
echo ============================================================
pause
