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

REM Fix MSVC compatibility issues in GPR source
echo Fixing MSVC compatibility issues...

REM 1. Create a patch file for CMakeLists.txt
echo Patching CMakeLists.txt for MSVC...
echo # MSVC Compatibility Patch > cmake_patch.txt
echo if(MSVC) >> cmake_patch.txt
echo     add_definitions(-Dfallthrough=) >> cmake_patch.txt
echo     add_definitions(-D__attribute__^(x^)=) >> cmake_patch.txt
echo else() >> cmake_patch.txt
echo     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99") >> cmake_patch.txt
echo endif() >> cmake_patch.txt

REM Apply the patch by prepending to CMakeLists.txt
if exist CMakeLists.txt.bak (
    echo CMakeLists.txt already patched
) else (
    copy CMakeLists.txt CMakeLists.txt.bak
    type cmake_patch.txt > CMakeLists.tmp
    type CMakeLists.txt >> CMakeLists.tmp
    move /y CMakeLists.tmp CMakeLists.txt
    echo CMakeLists.txt patched
)

REM 2. Directly patch the problematic files for MSVC compatibility
echo Patching files for MSVC compatibility...

REM First patch xmltok_impl.c to remove __attribute__((fallthrough)) usage
if not exist source\lib\expat_lib\xmltok_impl.c.bak (
    copy source\lib\expat_lib\xmltok_impl.c source\lib\expat_lib\xmltok_impl.c.bak
    REM Replace __attribute__ fallthrough with comment
    powershell -Command "$content = Get-Content 'source\lib\expat_lib\xmltok_impl.c.bak' -Raw; $content = $content -replace '__attribute__\s*\(\s*\(\s*fallthrough\s*\)\s*\)\s*;', '/* fallthrough */;'; Set-Content -Path 'source\lib\expat_lib\xmltok_impl.c' -Value $content"
    echo Patched xmltok_impl.c - removed __attribute__ fallthrough
)

REM Patch xmltok.c to also handle attribute and remove any remaining fallthrough
if not exist source\lib\expat_lib\xmltok.c.bak (
    copy source\lib\expat_lib\xmltok.c source\lib\expat_lib\xmltok.c.bak
    REM Replace __attribute__ fallthrough and bare fallthrough
    powershell -Command "$content = Get-Content 'source\lib\expat_lib\xmltok.c.bak' -Raw; $content = $content -replace '__attribute__\s*\(\s*\(\s*fallthrough\s*\)\s*\)\s*;', '/* fallthrough */;'; $content = $content -replace 'fallthrough;', '/* fallthrough */;'; Set-Content -Path 'source\lib\expat_lib\xmltok.c' -Value $content"
    echo Patched xmltok.c - removed fallthrough usage
)

REM Create a simpler compatibility header that we'll prepend to files
echo Creating MSVC compatibility definitions...
echo #ifdef _MSC_VER > msvc_compat.h
echo #undef __attribute__ >> msvc_compat.h
echo #define __attribute__(x) >> msvc_compat.h
echo #undef FASTCALL >> msvc_compat.h
echo #define FASTCALL __fastcall >> msvc_compat.h
echo #undef PTRFASTCALL >> msvc_compat.h
echo #define PTRFASTCALL __fastcall >> msvc_compat.h
echo #undef XMLCALL >> msvc_compat.h
echo #define XMLCALL __cdecl >> msvc_compat.h
echo #undef FALL_THROUGH >> msvc_compat.h
echo #define FALL_THROUGH >> msvc_compat.h
echo #undef fallthrough >> msvc_compat.h
echo #define fallthrough >> msvc_compat.h
echo #endif >> msvc_compat.h

REM Now prepend this header to the problematic files
if not exist source\lib\expat_lib\xmltok.h.bak (
    copy source\lib\expat_lib\xmltok.h source\lib\expat_lib\xmltok.h.bak
    type msvc_compat.h > source\lib\expat_lib\xmltok.h.tmp
    type source\lib\expat_lib\xmltok.h.bak >> source\lib\expat_lib\xmltok.h.tmp
    move /y source\lib\expat_lib\xmltok.h.tmp source\lib\expat_lib\xmltok.h
    echo Patched xmltok.h with MSVC compatibility
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