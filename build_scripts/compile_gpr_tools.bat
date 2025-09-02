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

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake -G "Visual Studio 16 2019" -A x64 ..
if %errorlevel% neq 0 (
    echo CMake configuration failed. Trying with MinGW...
    cmake -G "MinGW Makefiles" ..
    if %errorlevel% neq 0 (
        echo CMake configuration failed. Please ensure Visual Studio or MinGW is installed.
        exit /b 1
    )
    mingw32-make
) else (
    REM Build with MSBuild
    echo Building with Visual Studio...
    cmake --build . --config Release
)

REM Copy binary
if not exist "..\..\..\binaries\win32" mkdir "..\..\..\binaries\win32"
copy "source\app\gpr_tools\Release\gpr_tools.exe" "..\..\..\binaries\win32\gpr_tools.exe"
if %errorlevel% neq 0 (
    REM Try alternative location
    copy "source\app\gpr_tools\gpr_tools.exe" "..\..\..\binaries\win32\gpr_tools.exe"
)

cd ..\..\..

if exist "binaries\win32\gpr_tools.exe" (
    echo Success: Binary copied to binaries\win32\gpr_tools.exe
) else (
    echo Error: Failed to build or copy gpr_tools.exe
    exit /b 1
)

echo gpr_tools compiled successfully