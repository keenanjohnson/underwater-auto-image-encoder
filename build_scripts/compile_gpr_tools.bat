@echo off
REM Compile gpr_tools for Windows
REM Requires Visual Studio 2019 or later with CMake support

echo Compiling gpr_tools for Windows...

REM Check if temp/gpr exists
if not exist "temp\gpr" (
    echo Cloning GPR repository...
    mkdir temp
    cd temp
    git clone https://github.com/gopro/gpr.git
    cd ..
)

cd temp\gpr

REM Update to latest
git pull

REM Clean and create build directory
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)
mkdir build
cd build

REM Try different build methods
echo Configuring with CMake...

REM Check if running in CI
if defined CI (
    echo Running in CI, using Ninja...
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release
    if %errorlevel% neq 0 (
        echo Ninja configuration failed.
        exit /b 1
    )
    ninja
) else (
    REM Try default generator for local builds
    cmake .. -DCMAKE_BUILD_TYPE=Release
    if %errorlevel% neq 0 (
        echo Default CMake configuration failed. Trying Ninja...
        cd ..
        rmdir /s /q build
        mkdir build
        cd build
        cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release
        if %errorlevel% neq 0 (
            echo CMake configuration failed. Please ensure build tools are installed.
            exit /b 1
        )
        ninja
    ) else (
        REM Build with default generator
        echo Building with default generator...
        cmake --build . --config Release
    )
)

REM Copy binary - try different possible locations
if not exist "..\..\..\binaries\win32" mkdir "..\..\..\binaries\win32"

REM Try Release build location
if exist "source\app\gpr_tools\Release\gpr_tools.exe" (
    copy "source\app\gpr_tools\Release\gpr_tools.exe" "..\..\..\binaries\win32\gpr_tools.exe"
    goto :check_copy
)

REM Try Ninja build location
if exist "source\app\gpr_tools\gpr_tools.exe" (
    copy "source\app\gpr_tools\gpr_tools.exe" "..\..\..\binaries\win32\gpr_tools.exe"
    goto :check_copy
)

REM Try root build location (Ninja sometimes puts it here)
if exist "gpr_tools.exe" (
    copy "gpr_tools.exe" "..\..\..\binaries\win32\gpr_tools.exe"
    goto :check_copy
)

echo Warning: Could not find gpr_tools.exe in expected locations
exit /b 1

:check_copy

cd ..\..\..

if exist "binaries\win32\gpr_tools.exe" (
    echo Success: Binary copied to binaries\win32\gpr_tools.exe
) else (
    echo Error: Failed to build or copy gpr_tools.exe
    exit /b 1
)

echo gpr_tools compiled successfully