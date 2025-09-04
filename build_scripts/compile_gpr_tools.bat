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
        echo Git apply failed, applying comprehensive manual patches...
        goto :comprehensive_patch
    )
    echo Patch applied successfully
    goto :cmake_patch
) else (
    echo MSVC patch file not found, applying comprehensive manual patches...
    goto :comprehensive_patch
)

:comprehensive_patch
REM Apply comprehensive MSVC compatibility fixes by creating clean versions of problematic files
echo Applying comprehensive MSVC compatibility patches...

REM 1. Create a completely clean version of xmltok.h with MSVC compatibility
if not exist source\lib\expat_lib\xmltok.h.bak (
    copy source\lib\expat_lib\xmltok.h source\lib\expat_lib\xmltok.h.bak
    
    REM Create a clean, MSVC-compatible xmltok.h
    echo /* Copyright ^(c^) 1998, 1999 Thai Open Source Software Center Ltd > source\lib\expat_lib\xmltok.h
    echo    See the file COPYING for copying permission. >> source\lib\expat_lib\xmltok.h
    echo */ >> source\lib\expat_lib\xmltok.h
    echo. >> source\lib\expat_lib\xmltok.h
    echo /* MSVC compatibility defines */ >> source\lib\expat_lib\xmltok.h
    echo #ifdef _MSC_VER >> source\lib\expat_lib\xmltok.h
    echo #define fallthrough >> source\lib\expat_lib\xmltok.h
    echo #define __attribute__^(x^) >> source\lib\expat_lib\xmltok.h
    echo #define FASTCALL __fastcall >> source\lib\expat_lib\xmltok.h
    echo #define PTRFASTCALL __fastcall >> source\lib\expat_lib\xmltok.h
    echo #define XMLCALL __cdecl >> source\lib\expat_lib\xmltok.h
    echo #endif >> source\lib\expat_lib\xmltok.h
    echo. >> source\lib\expat_lib\xmltok.h
    
    REM Extract and append the actual header content, skipping first 4 lines (copyright + blank)
    powershell -Command "try { $lines = Get-Content 'source\lib\expat_lib\xmltok.h.bak'; $content = $lines[4..$($lines.Length-1)] -join \"`n\"; Add-Content -Path 'source\lib\expat_lib\xmltok.h' -Value $content } catch { Write-Host 'PowerShell header merge failed' }"
    echo Recreated xmltok.h with MSVC compatibility
)

REM 2. Use a comprehensive sed-like replacement for the C files
REM First, create a PowerShell script file for comprehensive cleaning
if not exist clean_expat.ps1 (
    echo # PowerShell script to clean expat files for MSVC compatibility > clean_expat.ps1
    echo param^([string]$FilePath^) >> clean_expat.ps1
    echo. >> clean_expat.ps1
    echo try { >> clean_expat.ps1
    echo     $content = Get-Content $FilePath -Raw >> clean_expat.ps1
    echo     if ^($content^) { >> clean_expat.ps1
    echo         # Remove all __attribute__ patterns comprehensively >> clean_expat.ps1
    echo         $content = $content -replace '__attribute__\s*\(\s*\([^)]*\)\s*\)', '' >> clean_expat.ps1
    echo         $content = $content -replace '__attribute__\s*\(\([^)]*\)\)', '' >> clean_expat.ps1  
    echo         $content = $content -replace '__attribute__\([^)]*\)', '' >> clean_expat.ps1
    echo         # Remove standalone fallthrough >> clean_expat.ps1
    echo         $content = $content -replace 'fallthrough\s*;', '/* fallthrough */;' >> clean_expat.ps1
    echo         $content = $content -replace '\s*fallthrough\s*$', ' /* fallthrough */' >> clean_expat.ps1
    echo         # Clean up any remaining attribute artifacts >> clean_expat.ps1
    echo         $content = $content -replace '\s*\(\s*\(\s*fallthrough\s*\)\s*\)\s*;', ' /* fallthrough */;' >> clean_expat.ps1
    echo         $content = $content -replace '\(\s*fallthrough\s*\)', '/* fallthrough */' >> clean_expat.ps1
    echo         # Write cleaned content >> clean_expat.ps1
    echo         [System.IO.File]::WriteAllText^($FilePath, $content^) >> clean_expat.ps1
    echo         Write-Host "Cleaned $FilePath successfully" >> clean_expat.ps1
    echo     } >> clean_expat.ps1
    echo } catch { >> clean_expat.ps1
    echo     Write-Host "Error cleaning $FilePath`: $_" >> clean_expat.ps1
    echo } >> clean_expat.ps1
)

REM 3. Apply comprehensive cleaning to the problematic files
if not exist source\lib\expat_lib\xmltok_impl.c.bak (
    copy source\lib\expat_lib\xmltok_impl.c source\lib\expat_lib\xmltok_impl.c.bak
    powershell -ExecutionPolicy Bypass -File clean_expat.ps1 -FilePath "source\lib\expat_lib\xmltok_impl.c"
    echo Cleaned xmltok_impl.c
)

if not exist source\lib\expat_lib\xmltok.c.bak (
    copy source\lib\expat_lib\xmltok.c source\lib\expat_lib\xmltok.c.bak
    powershell -ExecutionPolicy Bypass -File clean_expat.ps1 -FilePath "source\lib\expat_lib\xmltok.c"
    echo Cleaned xmltok.c
)

REM Clean up the temporary PowerShell script
del clean_expat.ps1

:cmake_patch
REM 4. Patch CMakeLists.txt for MSVC compatibility with more comprehensive flags
if not exist CMakeLists.txt.bak (
    copy CMakeLists.txt CMakeLists.txt.bak
    
    REM Create comprehensive MSVC-compatible CMakeLists.txt
    echo # MSVC Compatibility Settings - Must be at top > cmake_msvc.txt
    echo if^(MSVC^) >> cmake_msvc.txt
    echo     # Remove problematic flags for MSVC >> cmake_msvc.txt
    echo     string^(REPLACE "-std=c99" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}"^) >> cmake_msvc.txt
    echo     string^(REPLACE "-std=c++11" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}"^) >> cmake_msvc.txt
    echo     # Add comprehensive MSVC compatibility definitions >> cmake_msvc.txt
    echo     add_definitions^(-D_CRT_SECURE_NO_WARNINGS^) >> cmake_msvc.txt
    echo     add_definitions^(-Dfallthrough=^) >> cmake_msvc.txt
    echo     add_definitions^(-D__attribute__^(x^)=^) >> cmake_msvc.txt
    echo     add_definitions^(-DFASTCALL=__fastcall^) >> cmake_msvc.txt
    echo     add_definitions^(-DPTRFASTCALL=__fastcall^) >> cmake_msvc.txt
    echo     add_definitions^(-DXMLCALL=__cdecl^) >> cmake_msvc.txt
    echo     # Force exception handling for C++ >> cmake_msvc.txt
    echo     set^(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc"^) >> cmake_msvc.txt
    echo     # Add additional MSVC-specific flags >> cmake_msvc.txt
    echo     set^(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4068 /wd4244 /wd4996"^) >> cmake_msvc.txt
    echo     set^(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068 /wd4244 /wd4996"^) >> cmake_msvc.txt
    echo     message^(STATUS "Applied MSVC compatibility settings"^) >> cmake_msvc.txt
    echo else^(^) >> cmake_msvc.txt
    echo     set^(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99"^) >> cmake_msvc.txt
    echo     set^(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11"^) >> cmake_msvc.txt
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