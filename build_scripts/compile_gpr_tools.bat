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

REM Apply MSVC compatibility patch
echo Applying MSVC compatibility patches...

REM Check if we have the patch file and apply it
if exist "..\..\msvc_compatibility.patch" (
    echo Applying existing MSVC compatibility patch...
    git apply "..\..\msvc_compatibility.patch" 2>nul || (
        echo Git apply failed, applying patch manually...
        goto :manual_patch
    )
    echo Patch applied successfully
    goto :cmake_patch
) else (
    echo MSVC patch file not found, applying manual patches...
    goto :manual_patch
)

:manual_patch
REM Apply comprehensive MSVC compatibility fixes
echo Applying manual MSVC compatibility patches...

REM 1. Patch xmltok.h - add fallthrough definition at the top
if not exist source\lib\expat_lib\xmltok.h.bak (
    copy source\lib\expat_lib\xmltok.h source\lib\expat_lib\xmltok.h.bak
    
    REM Create the patched header content
    echo /* Copyright ^(c^) 1998, 1999 Thai Open Source Software Center Ltd > xmltok_new.h
    echo    See the file COPYING for copying permission. >> xmltok_new.h
    echo */ >> xmltok_new.h
    echo. >> xmltok_new.h
    echo /* MSVC compatibility - define GNU C extensions as empty */ >> xmltok_new.h
    echo #ifdef _MSC_VER >> xmltok_new.h
    echo #ifndef fallthrough >> xmltok_new.h
    echo #define fallthrough >> xmltok_new.h
    echo #endif >> xmltok_new.h
    echo #ifndef __attribute__ >> xmltok_new.h
    echo #define __attribute__^(x^) >> xmltok_new.h
    echo #endif >> xmltok_new.h
    echo #endif >> xmltok_new.h
    echo. >> xmltok_new.h
    
    REM Skip the copyright lines from original and append the rest
    more +3 source\lib\expat_lib\xmltok.h.bak >> xmltok_new.h
    move /y xmltok_new.h source\lib\expat_lib\xmltok.h
    echo Patched xmltok.h with MSVC compatibility
)

REM 2. Patch xmltok_impl.c to remove __attribute__ usage
if not exist source\lib\expat_lib\xmltok_impl.c.bak (
    copy source\lib\expat_lib\xmltok_impl.c source\lib\expat_lib\xmltok_impl.c.bak
    
    REM Use more robust PowerShell replacement
    powershell -Command "try { $content = Get-Content 'source\lib\expat_lib\xmltok_impl.c.bak' -Raw; if ($content) { $content = $content -replace '__attribute__\s*\(\s*\([^)]*\)\s*\)', ''; $content = $content -replace 'fallthrough\s*;', '/* fallthrough */;'; [System.IO.File]::WriteAllText('source\lib\expat_lib\xmltok_impl.c', $content) } } catch { Write-Host 'PowerShell patch failed' }"
    echo Patched xmltok_impl.c
)

REM 3. Patch xmltok.c to remove __attribute__ usage  
if not exist source\lib\expat_lib\xmltok.c.bak (
    copy source\lib\expat_lib\xmltok.c source\lib\expat_lib\xmltok.c.bak
    
    REM Use more robust PowerShell replacement
    powershell -Command "try { $content = Get-Content 'source\lib\expat_lib\xmltok.c.bak' -Raw; if ($content) { $content = $content -replace '__attribute__\s*\(\s*\([^)]*\)\s*\)', ''; $content = $content -replace 'fallthrough\s*;', '/* fallthrough */;'; [System.IO.File]::WriteAllText('source\lib\expat_lib\xmltok.c', $content) } } catch { Write-Host 'PowerShell patch failed' }"
    echo Patched xmltok.c
)

:cmake_patch
REM 4. Patch CMakeLists.txt for MSVC compatibility
if not exist CMakeLists.txt.bak (
    copy CMakeLists.txt CMakeLists.txt.bak
    
    REM Create MSVC-compatible CMakeLists.txt
    echo # MSVC Compatibility Settings > cmake_msvc.txt
    echo if^(MSVC^) >> cmake_msvc.txt
    echo     # Remove problematic C99 flag for MSVC >> cmake_msvc.txt
    echo     string^(REPLACE "-std=c99" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}"^) >> cmake_msvc.txt
    echo     # Add MSVC-specific definitions >> cmake_msvc.txt
    echo     add_definitions^(-D_CRT_SECURE_NO_WARNINGS^) >> cmake_msvc.txt
    echo     add_definitions^(-Dfallthrough=^) >> cmake_msvc.txt
    echo     add_definitions^(-D__attribute__^(x^)=^) >> cmake_msvc.txt
    echo     # Enable exception handling >> cmake_msvc.txt
    echo     set^(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc"^) >> cmake_msvc.txt
    echo else^(^) >> cmake_msvc.txt
    echo     set^(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99"^) >> cmake_msvc.txt
    echo endif^(^) >> cmake_msvc.txt
    echo. >> cmake_msvc.txt
    
    REM Prepend to original CMakeLists.txt
    type cmake_msvc.txt > CMakeLists_new.txt
    type CMakeLists.txt.bak >> CMakeLists_new.txt
    move /y CMakeLists_new.txt CMakeLists.txt
    del cmake_msvc.txt
    echo Patched CMakeLists.txt for MSVC compatibility
)

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
    REM For MSVC, add comprehensive compatibility flags
    set "CMAKE_C_FLAGS=/D_CRT_SECURE_NO_WARNINGS /Dfallthrough= /D__attribute__(x)="
    set "CMAKE_CXX_FLAGS=/D_CRT_SECURE_NO_WARNINGS /Dfallthrough= /D__attribute__(x)= /EHsc"
    
    REM Try Ninja first (faster)
    where ninja >nul 2>nul
    if %errorlevel% equ 0 (
        echo Using Ninja generator with MSVC compatibility flags...
        cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="%CMAKE_C_FLAGS%" -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS%" -DCMAKE_POLICY_DEFAULT_CMP0054=NEW
        if %errorlevel% equ 0 (
            echo Building with Ninja...
            ninja -j4 gpr_tools
            if %errorlevel% neq 0 (
                echo Ninja build failed, trying full build...
                ninja
            )
        ) else (
            echo Ninja configuration failed, trying Visual Studio generator...
            cd ..
            rmdir /s /q build
            mkdir build
            cd build
            cmake -G "Visual Studio 17 2022" -A x64 .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="%CMAKE_C_FLAGS%" -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS%" -DCMAKE_POLICY_DEFAULT_CMP0054=NEW
            if %errorlevel% equ 0 (
                echo Building with Visual Studio...
                cmake --build . --config Release --target gpr_tools --parallel 4
                if %errorlevel% neq 0 (
                    echo Target build failed, trying full build...
                    cmake --build . --config Release --parallel 4
                )
            )
        )
    ) else (
        echo Ninja not found, using Visual Studio generator...
        cmake -G "Visual Studio 17 2022" -A x64 .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="%CMAKE_C_FLAGS%" -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS%" -DCMAKE_POLICY_DEFAULT_CMP0054=NEW
        if %errorlevel% equ 0 (
            echo Building with Visual Studio...
            cmake --build . --config Release --target gpr_tools --parallel 4
            if %errorlevel% neq 0 (
                echo Target build failed, trying full build...
                cmake --build . --config Release --parallel 4
            )
        )
    )
) else (
    REM Try with MSVC compatibility flags for local builds  
    set "CMAKE_C_FLAGS=/D_CRT_SECURE_NO_WARNINGS /Dfallthrough= /D__attribute__(x)="
    set "CMAKE_CXX_FLAGS=/D_CRT_SECURE_NO_WARNINGS /Dfallthrough= /D__attribute__(x)= /EHsc"
    
    echo Trying Visual Studio generator with MSVC compatibility...
    cmake -G "Visual Studio 17 2022" -A x64 .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="%CMAKE_C_FLAGS%" -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS%" -DCMAKE_POLICY_DEFAULT_CMP0054=NEW
    if %errorlevel% neq 0 (
        echo VS2022 not found, trying VS2019...
        cmake -G "Visual Studio 16 2019" -A x64 .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="%CMAKE_C_FLAGS%" -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS%" -DCMAKE_POLICY_DEFAULT_CMP0054=NEW
        if %errorlevel% neq 0 (
            echo Visual Studio not found. Trying default generator...
            cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="%CMAKE_C_FLAGS%" -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS%" -DCMAKE_POLICY_DEFAULT_CMP0054=NEW
            if %errorlevel% neq 0 (
                echo CMake configuration failed. Please ensure Visual Studio or Build Tools are installed.
                exit /b 1
            )
        )
    )
    REM Build with configured generator
    echo Building gpr_tools...
    cmake --build . --config Release --target gpr_tools --parallel 4
    if %errorlevel% neq 0 (
        echo Target build failed, trying full build...
        cmake --build . --config Release --parallel 4
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