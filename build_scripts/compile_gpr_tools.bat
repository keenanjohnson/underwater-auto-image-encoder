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

REM Update to latest and reset any changes  
git reset --hard HEAD
git pull

REM Fix MSVC compatibility issues using the simple patch approach
echo Fixing MSVC compatibility issues...

REM Apply the patch to xmltok.h - this is the key file that needs patching
echo Patching CMakeLists.txt for MSVC...
if not exist CMakeLists.txt.bak (
    copy CMakeLists.txt CMakeLists.txt.bak
    REM Create a simple patch that handles MSVC C99 standard flag
    (
        echo # MSVC Compatibility - don't use -std=c99 flag for MSVC
        echo if^(MSVC^)
        echo     # MSVC doesn't support -std=c99 flag, and already defaults to C99-compatible mode
        echo     add_definitions^(-D_CRT_SECURE_NO_WARNINGS^)
        echo else^(^)
        echo     set^(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99"^)
        echo endif^(^)
        echo.
        type CMakeLists.txt.bak
    ) > CMakeLists.txt
    echo CMakeLists.txt patched
)

REM Apply the xmltok.h patch to define fallthrough for MSVC
echo Creating MSVC compatibility header...
if not exist source\lib\expat_lib\xmltok.h.bak (
    copy source\lib\expat_lib\xmltok.h source\lib\expat_lib\xmltok.h.bak
    REM Apply the patch inline - add fallthrough definition after the copyright notice
    (
        echo /* Copyright ^(c^) 1998, 1999 Thai Open Source Software Center Ltd
        echo    See the file COPYING for copying permission.
        echo */
        echo.
        echo /* MSVC compatibility - define fallthrough as empty */
        echo #ifdef _MSC_VER
        echo #define fallthrough
        echo #endif
        echo.
        more +3 source\lib\expat_lib\xmltok.h.bak
    ) > source\lib\expat_lib\xmltok.h
    echo Patched xmltok.h
)

REM Clean and create build directory
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)
mkdir build
cd build

REM Configure and build with CMake
echo Configuring with CMake...

REM Check if running in CI
if defined CI (
    echo Running in CI environment...
    
    REM Try Ninja first (faster) with proper MSVC flags
    where ninja >nul 2>nul
    if %errorlevel% equ 0 (
        echo Using Ninja generator with MSVC compatibility flags...
        cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="/D_CRT_SECURE_NO_WARNINGS" -DCMAKE_CXX_FLAGS="/D_CRT_SECURE_NO_WARNINGS /EHsc"
        if %errorlevel% equ 0 (
            ninja
        ) else (
            echo Ninja configuration failed, trying Visual Studio generator...
            cd ..
            rmdir /s /q build
            mkdir build
            cd build
            cmake -G "Visual Studio 17 2022" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
            if %errorlevel% equ 0 (
                cmake --build . --config Release --target gpr_tools
            )
        )
    ) else (
        echo Ninja not found, using Visual Studio generator...
        cmake -G "Visual Studio 17 2022" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
        if %errorlevel% equ 0 (
            cmake --build . --config Release --target gpr_tools
        )
    )
) else (
    REM For local builds
    echo Trying Visual Studio generator with MSVC compatibility...
    cmake -G "Visual Studio 17 2022" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
    if %errorlevel% neq 0 (
        echo VS2022 not found, trying VS2019...
        cmake -G "Visual Studio 16 2019" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
        if %errorlevel% neq 0 (
            echo Visual Studio not found. Trying default generator...
            cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
            if %errorlevel% neq 0 (
                echo CMake configuration failed. Please ensure Visual Studio or Build Tools are installed.
                exit /b 1
            )
        )
    )
    REM Build with configured generator
    echo Building gpr_tools...
    cmake --build . --config Release --target gpr_tools
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