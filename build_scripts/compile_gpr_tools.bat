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

REM Apply MSVC compatibility fixes using git patch approach with comprehensive fallbacks
echo Fixing MSVC compatibility issues...

REM Create a comprehensive patch for MSVC compatibility
echo Creating comprehensive MSVC compatibility patch...
(
echo From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001
echo From: Build Script ^<build@example.com^>
echo Date: Wed, 1 Jan 2025 00:00:00 +0000
echo Subject: [PATCH] Add comprehensive MSVC compatibility fixes
echo.
echo diff --git a/CMakeLists.txt b/CMakeLists.txt
echo index 1234567..abcdefg 100644
echo --- a/CMakeLists.txt
echo +++ b/CMakeLists.txt
echo @@ -1,3 +1,12 @@
echo +# MSVC Compatibility
echo +if^(MSVC^)
echo +    add_definitions^(-D_CRT_SECURE_NO_WARNINGS^)
echo +    add_definitions^(-Dfallthrough=^)
echo +    add_definitions^(-D__attribute__^(x^)=^)
echo +    # MSVC doesn't need -std=c99
echo +else^(^)
echo +    set^(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99"^)
echo +endif^(^)
echo +
echo  cmake_minimum_required^(VERSION 3.5 FATAL_ERROR^)
echo.
echo  if^(POLICY CMP0048^)
) > msvc_compat.patch

REM Apply the patch
git apply --ignore-whitespace msvc_compat.patch
if %errorlevel% neq 0 (
    echo Git patch failed, trying comprehensive manual approach...
    
    REM 1. Patch CMakeLists.txt with comprehensive MSVC compatibility
    if not exist CMakeLists.txt.bak (
        copy CMakeLists.txt CMakeLists.txt.bak
        (
            echo # MSVC Compatibility
            echo if^(MSVC^)
            echo     add_definitions^(-D_CRT_SECURE_NO_WARNINGS^)
            echo     add_definitions^(-Dfallthrough=^)
            echo     add_definitions^(-D__attribute__^(x^)=^)
            echo     # MSVC doesn't need -std=c99
            echo else^(^)
            echo     set^(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99"^)
            echo endif^(^)
            echo.
            type CMakeLists.txt.bak
        ) > CMakeLists.txt
        echo CMakeLists.txt patched with comprehensive MSVC compatibility
    )
    
    REM 2. Patch xmltok_impl.c to remove __attribute__((fallthrough)) usage
    if not exist source\lib\expat_lib\xmltok_impl.c.bak (
        copy source\lib\expat_lib\xmltok_impl.c source\lib\expat_lib\xmltok_impl.c.bak
        REM Replace __attribute__((fallthrough)); with /* fallthrough */
        powershell -Command "(Get-Content 'source\lib\expat_lib\xmltok_impl.c.bak' -Raw) -replace '__attribute__\s*\(\s*\(\s*fallthrough\s*\)\s*\)\s*;', '/* fallthrough */;' | Set-Content 'source\lib\expat_lib\xmltok_impl.c'"
        echo Patched xmltok_impl.c - removed __attribute__((fallthrough))
    )

    REM 3. Patch xmltok.c to handle attribute and remove any remaining fallthrough
    if not exist source\lib\expat_lib\xmltok.c.bak (
        copy source\lib\expat_lib\xmltok.c source\lib\expat_lib\xmltok.c.bak
        REM Replace __attribute__((fallthrough)); and bare fallthrough;
        powershell -Command "(Get-Content 'source\lib\expat_lib\xmltok.c.bak' -Raw) -replace '__attribute__\s*\(\s*\(\s*fallthrough\s*\)\s*\)\s*;', '/* fallthrough */;' -replace '(?<!/)fallthrough;', '/* fallthrough */;' | Set-Content 'source\lib\expat_lib\xmltok.c'"
        echo Patched xmltok.c - removed fallthrough usage
    )

    REM 4. Patch internal.h directly to handle __attribute__ macros
    if not exist source\lib\expat_lib\internal.h.bak (
        copy source\lib\expat_lib\internal.h source\lib\expat_lib\internal.h.bak
        powershell -Command "(Get-Content 'source\lib\expat_lib\internal.h.bak') -replace '#define FASTCALL __attribute__\(\(stdcall, regparm\(3\)\)\)', '#ifdef _MSC_VER`n#define FASTCALL __fastcall`n#else`n#define FASTCALL __attribute__((stdcall, regparm(3)))`n#endif' -replace '#define FASTCALL __attribute__\(\(regparm\(3\)\)\)', '#ifdef _MSC_VER`n#define FASTCALL __fastcall`n#else`n#define FASTCALL __attribute__((regparm(3)))`n#endif' -replace '#define PTRFASTCALL __attribute__\(\(regparm\(3\)\)\)', '#ifdef _MSC_VER`n#define PTRFASTCALL __fastcall`n#else`n#define PTRFASTCALL __attribute__((regparm(3)))`n#endif' -replace '#define FALL_THROUGH __attribute__ \(\(fallthrough\)\)', '#ifdef _MSC_VER`n#define FALL_THROUGH`n#else`n#define FALL_THROUGH __attribute__ ((fallthrough))`n#endif' | Set-Content 'source\lib\expat_lib\internal.h'"
        echo Patched internal.h with MSVC compatibility
    )

    REM 5. Patch expat_external.h to handle XMLCALL
    if not exist source\lib\expat_lib\expat_external.h.bak (
        copy source\lib\expat_lib\expat_external.h source\lib\expat_lib\expat_external.h.bak
        powershell -Command "(Get-Content 'source\lib\expat_lib\expat_external.h.bak') -replace '#define XMLCALL __attribute__\(\(cdecl\)\)', '#ifdef _MSC_VER`n#define XMLCALL __cdecl`n#else`n#define XMLCALL __attribute__((cdecl))`n#endif' | Set-Content 'source\lib\expat_lib\expat_external.h'"
        echo Patched expat_external.h with MSVC compatibility
    )
    
    echo Comprehensive MSVC compatibility patches applied manually
) else (
    echo MSVC compatibility patch applied successfully via git
)

REM Clean and create build directory
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)
mkdir build
cd build

echo Configuring with CMake...

REM Check if running in CI
if defined CI (
    echo Running in CI environment...
    REM For MSVC, disable some warnings and add compatibility defines
    set "CMAKE_C_FLAGS=/D_CRT_SECURE_NO_WARNINGS /D_MSVC_COMPAT"
    set "CMAKE_CXX_FLAGS=/D_CRT_SECURE_NO_WARNINGS /D_MSVC_COMPAT /EHsc"
    
    REM Try Ninja first (faster)
    where ninja >nul 2>nul
    if %errorlevel% equ 0 (
        echo Using Ninja generator with MSVC compatibility flags...
        cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="%CMAKE_C_FLAGS%" -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS%"
        if %errorlevel% equ 0 (
            ninja gpr_tools
        ) else (
            echo Ninja configuration failed, trying Visual Studio generator...
            cd ..
            rmdir /s /q build
            mkdir build
            cd build
            cmake -G "Visual Studio 17 2022" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="%CMAKE_C_FLAGS%" -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS%"
            if %errorlevel% equ 0 (
                cmake --build . --config Release --target gpr_tools
            )
        )
    ) else (
        echo Ninja not found, using Visual Studio generator...
        cmake -G "Visual Studio 17 2022" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="%CMAKE_C_FLAGS%" -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS%"
        if %errorlevel% equ 0 (
            cmake --build . --config Release --target gpr_tools
        )
    )
) else (
    REM Try with MSVC compatibility flags for local builds  
    set "CMAKE_C_FLAGS=/D_CRT_SECURE_NO_WARNINGS /D_MSVC_COMPAT"
    set "CMAKE_CXX_FLAGS=/D_CRT_SECURE_NO_WARNINGS /D_MSVC_COMPAT /EHsc"
    
    echo Trying Visual Studio generator with MSVC compatibility...
    cmake -G "Visual Studio 17 2022" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="%CMAKE_C_FLAGS%" -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS%"
    if %errorlevel% neq 0 (
        echo VS2022 not found, trying VS2019...
        cmake -G "Visual Studio 16 2019" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="%CMAKE_C_FLAGS%" -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS%"
        if %errorlevel% neq 0 (
            echo Visual Studio not found. Trying default generator...
            cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="%CMAKE_C_FLAGS%" -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS%"
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