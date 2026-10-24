param(
    [string]$AndroidSdk = "D:/projects/phoneproject/.tools/android-sdk",
    [string]$AndroidNdk = "D:/projects/phoneproject/.tools/android-sdk/ndk/27.1.12297006",
    [string]$GgmlSource = "D:/op13-root/src/ggml-hexagon-091b6a7",
    [string]$Zig = "D:/op13-root/tools/zig-0.17.0/zig.exe",
    [string]$AndroidLibc = "D:/op13-root/tools/android-ndk-r30-zig-libc-api36.txt",
    [string]$QairtSdk = "D:/op13-root/tools/qairt-sdk/qairt/2.50.0.260828",
    [ValidateSet("debug", "release")][string]$Configuration = "debug"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Mobile = Join-Path $Repo "app/mobile"
$Cmake = Join-Path $AndroidSdk "cmake/3.31.6/bin/cmake.exe"
$Ninja = Join-Path $AndroidSdk "cmake/3.31.6/bin/ninja.exe"
$Glslc = Join-Path $AndroidNdk "shader-tools/windows-x86_64/glslc.exe"
$VulkanHeaders = "D:/op13-root/tools/Vulkan-Headers-1.4.335.0"
$VulkanCommit = "2fa203425eb4af9dfc6b03f97ef72b0b5bcb8350"
$VulkanBuild = Join-Path $GgmlSource "build-android-vulkan-ndk27"
$SpirvHeaders = Join-Path $AndroidNdk "sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include"
$Prefix = Join-Path $Repo "zig-out-android-vulkan"
$Runtime = Join-Path $Mobile "modules/tensorfold-local-runtime/android/src/main/jniLibs/arm64-v8a"

foreach ($Path in @($Cmake, $Ninja, $Glslc, $AndroidLibc, $Zig)) {
    if (-not (Test-Path $Path)) { throw "Missing Android build prerequisite: $Path" }
}
if (-not (Test-Path $VulkanHeaders)) {
    git clone --depth 1 --branch vulkan-sdk-1.4.335.0 https://github.com/KhronosGroup/Vulkan-Headers.git $VulkanHeaders
    if ($LASTEXITCODE -ne 0) { throw "Could not fetch the pinned Vulkan headers." }
}
$ActualVulkanCommit = (git -C $VulkanHeaders rev-parse HEAD).Trim()
if ($ActualVulkanCommit -ne $VulkanCommit) { throw "Unexpected Vulkan-Headers revision: $ActualVulkanCommit" }

New-Item -ItemType Directory -Force "$VulkanBuild/spirv-headers" | Out-Null
@"
set(SPIRV-Headers_FOUND TRUE)
set(SPIRV-Headers_INCLUDE_DIR "$($SpirvHeaders.Replace('\', '/'))")
include_directories("`${SPIRV-Headers_INCLUDE_DIR}")
"@ | Set-Content -Encoding ascii "$VulkanBuild/spirv-headers/SPIRV-HeadersConfig.cmake"
@"
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS -O2)
set(CMAKE_CXX_FLAGS -O2)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
set(CMAKE_C_COMPILER "$($Zig.Replace('\', '/'))")
set(CMAKE_C_COMPILER_ARG1 "cc")
set(CMAKE_CXX_COMPILER "$($Zig.Replace('\', '/'))")
set(CMAKE_CXX_COMPILER_ARG1 "c++")
set(CMAKE_MAKE_PROGRAM "$($Ninja.Replace('\', '/'))" CACHE FILEPATH "" FORCE)
"@ | Set-Content -Encoding ascii "$VulkanBuild/host-zig-toolchain.cmake"

$Configure = @(
    "-S", $GgmlSource, "-B", $VulkanBuild, "-G", "Ninja",
    "-DCMAKE_MAKE_PROGRAM=$Ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$AndroidNdk/build/cmake/android.toolchain.cmake",
    "-DCMAKE_CXX_FLAGS=-I$VulkanHeaders/include",
    "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-29", "-DANDROID_STL=c++_shared",
    "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON",
    "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF", "-DGGML_LLAMAFILE=OFF",
    "-DGGML_VULKAN=ON", "-DGGML_BACKEND_DL=OFF",
    "-DGGML_VULKAN_SHADERS_GEN_TOOLCHAIN=$VulkanBuild/host-zig-toolchain.cmake",
    "-DBUILD_SHARED_LIBS=ON", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF",
    "-DVulkan_GLSLC_EXECUTABLE=$Glslc", "-DSPIRV-Headers_DIR=$VulkanBuild/spirv-headers"
)
& $Cmake @Configure
if ($LASTEXITCODE -ne 0) { throw "Android Vulkan configuration failed." }
& $Cmake --build $VulkanBuild --target ggml-vulkan ggml-cpu -j 12
if ($LASTEXITCODE -ne 0) { throw "Android Vulkan build failed." }

$ZigCommon = @(
    "-Dtarget=aarch64-linux-android", "-Doptimize=fast", "-Dandroid-libc=$AndroidLibc",
    "-Dggml-include=$GgmlSource/ggml/include", "-Dggml-lib=$VulkanBuild/bin",
    "-Dggml-vulkan-lib=$VulkanBuild/bin", "--prefix", $Prefix
)
Push-Location $Repo
try {
    & $Zig build qualcomm-exl3-forward @ZigCommon
    if ($LASTEXITCODE -ne 0) { throw "2B Android engine build failed." }
    & $Zig build qualcomm-exl3-forward-4b @ZigCommon
    if ($LASTEXITCODE -ne 0) { throw "4B Android engine build failed." }
    & $Zig build qualcomm-exl3-forward-35b-a3b @ZigCommon
    if ($LASTEXITCODE -ne 0) { throw "35B-A3B Android engine build failed." }
    & $Zig build qualcomm-openai -Dtarget=aarch64-linux-android -Doptimize=fast "-Dandroid-libc=$AndroidLibc" --prefix $Prefix
    if ($LASTEXITCODE -ne 0) { throw "Android control-plane build failed." }
} finally { Pop-Location }

New-Item -ItemType Directory -Force $Runtime | Out-Null
Get-ChildItem $Runtime -Filter "*.so" -ErrorAction SilentlyContinue | Remove-Item -Force
$Files = @{
    "$Prefix/bin/tensorfold-qualcomm-openai" = "libtensorfold_openai.so"
    "$Prefix/bin/tensorfold-qualcomm-exl3-forward" = "libtensorfold_engine_2b.so"
    "$Prefix/bin/tensorfold-qualcomm-exl3-forward-4b" = "libtensorfold_engine_4b.so"
    "$Prefix/bin/tensorfold-qualcomm-exl3-forward-35b-a3b" = "libtensorfold_engine_moe.so"
    "$VulkanBuild/bin/libggml-base.so" = "libggml-base.so"
    "$VulkanBuild/bin/libggml-cpu.so" = "libggml-cpu.so"
    "$VulkanBuild/bin/libggml-vulkan.so" = "libggml-vulkan.so"
    "$AndroidNdk/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" = "libc++_shared.so"
}
foreach ($Entry in $Files.GetEnumerator()) {
    if (-not (Test-Path $Entry.Key)) { throw "Missing runtime artifact: $($Entry.Key)" }
    Copy-Item -Force $Entry.Key (Join-Path $Runtime $Entry.Value)
}

$QnnFiles = @{
    "$QairtSdk/bin/aarch64-android/qnn-platform-validator" = "libqnn_platform_validator.so"
    "$QairtSdk/lib/aarch64-android/libQnnHtp.so" = "libQnnHtp.so"
    "$QairtSdk/lib/aarch64-android/libQnnSystem.so" = "libQnnSystem.so"
    "$QairtSdk/lib/aarch64-android/libQnnHtpV79Stub.so" = "libQnnHtpV79Stub.so"
    "$QairtSdk/lib/aarch64-android/libQnnHtpV79CalculatorStub.so" = "libQnnHtpV79CalculatorStub.so"
    "$QairtSdk/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" = "libQnnHtpV79Skel.so"
    "$QairtSdk/lib/hexagon-v79/unsigned/libCalculator_skel.so" = "libCalculator_skel.so"
}
foreach ($Entry in $QnnFiles.GetEnumerator()) {
    if (-not (Test-Path $Entry.Key)) { throw "Missing QAIRT runtime artifact: $($Entry.Key)" }
    Copy-Item -Force $Entry.Key (Join-Path $Runtime $Entry.Value)
}
$Clang = Join-Path $AndroidNdk "toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
$QnnProbe = Join-Path $Mobile "modules/tensorfold-local-runtime/native/qnn_probe.cpp"
& $Clang "--target=aarch64-linux-android31" -shared -fPIC -O2 -Wl,-soname,libtensorfold_qnn_probe.so $QnnProbe -ldl -llog -o (Join-Path $Runtime "libtensorfold_qnn_probe.so")
if ($LASTEXITCODE -ne 0) { throw "TensorFold QNN app-UID probe build failed." }
New-Item -ItemType Directory -Force "$Mobile/assets" | Out-Null
Copy-Item -Force "$QairtSdk/NOTICE.txt" "$Mobile/assets/QAIRT_NOTICE.txt"

Push-Location $Mobile
try {
    npm ci
    if ($LASTEXITCODE -ne 0) { throw "Mobile dependency installation failed." }
    npx expo prebuild --platform android --clean --no-install
    if ($LASTEXITCODE -ne 0) { throw "Expo Android generation failed." }
    $env:ANDROID_HOME = $AndroidSdk
    $env:ANDROID_SDK_ROOT = $AndroidSdk
    $GradleTask = if ($Configuration -eq "release") { ":app:assembleRelease" } else { ":app:assembleDebug" }
    & "./android/gradlew.bat" -p android $GradleTask ":tensorfold-local-runtime:testDebugUnitTest"
    if ($LASTEXITCODE -ne 0) { throw "TensorFold Android app build failed." }
} finally { Pop-Location }

Get-FileHash "$Mobile/android/app/build/outputs/apk/$Configuration/app-$Configuration.apk"
