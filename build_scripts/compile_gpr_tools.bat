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
    echo Running in CI environment...
    REM Try Ninja first (faster)
    where ninja >nul 2>nul
    if %errorlevel% equ 0 (
        echo Using Ninja generator...
        cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release
        if %errorlevel% equ 0 (
            ninja gpr_tools
        ) else (
            echo Ninja configuration failed, trying default generator...
            cd ..
            rmdir /s /q build
            mkdir build
            cd build
            cmake .. -DCMAKE_BUILD_TYPE=Release
            if %errorlevel% equ 0 (
                cmake --build . --config Release --target gpr_tools
            )
        )
    ) else (
        echo Ninja not found, using default generator...
        cmake .. -DCMAKE_BUILD_TYPE=Release
        if %errorlevel% equ 0 (
            cmake --build . --config Release --target gpr_tools
        )
    )
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

REM Search for gpr_tools.exe in all possible locations
echo Searching for gpr_tools.exe...

REM Common build output locations
set LOCATIONS=source\app\gpr_tools\Release\gpr_tools.exe source\app\gpr_tools\gpr_tools.exe Release\gpr_tools.exe gpr_tools.exe bin\gpr_tools.exe bin\Release\gpr_tools.exe

for %%L in (%LOCATIONS%) do (
    if exist "%%L" (
        echo Found gpr_tools.exe at: %%L
        copy "%%L" "..\..\..\binaries\win32\gpr_tools.exe"
        goto :check_copy
    )
)

REM If still not found, search recursively
echo Searching recursively for gpr_tools.exe...
for /r . %%i in (gpr_tools.exe) do (
    if exist "%%i" (
        echo Found gpr_tools.exe at: %%i
        copy "%%i" "..\..\..\binaries\win32\gpr_tools.exe"
        goto :check_copy
    )
)

echo ERROR: Could not find gpr_tools.exe in any location
dir /s /b *.exe
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