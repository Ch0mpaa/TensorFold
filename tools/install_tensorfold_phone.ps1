param(
    [string]$Adb = "D:/projects/phoneproject/.tools/android-sdk/platform-tools/adb.exe",
    [string]$Serial = "e1db9f1d",
    [switch]$Start
)

$ErrorActionPreference = "Stop"
$Root = "/data/local/tmp/exl3-full-power"
$ModelDir = "/data/local/tmp/tensorfold-models/Qwen3.5-4B-exl3-4.00bpw"
$Protected = @{
    "$Root/tensorfold-qualcomm-exl3-forward-4b-mlperf-corpus-poc" = "6917588a7de1b51464822628dce095ad6eb7712db7081b0ec2150b37cb740144"
    "/data/local/tmp/tensorfold-mtp-poc/tensorfold-qualcomm-exl3-forward-4b-live-mtp" = "99eaeca94fb1c47046ab5e3cbf2c978a05e9eb97ca4342633fcf1c1c81e9a885"
}

function Invoke-Adb([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments) {
    & $Adb -s $Serial @Arguments
    if ($LASTEXITCODE -ne 0) { throw "adb exited $LASTEXITCODE" }
}

function Assert-Protected {
    foreach ($Entry in $Protected.GetEnumerator()) {
        $Actual = (Invoke-Adb shell "su -c 'sha256sum $($Entry.Key)'" | Select-Object -First 1).Split()[0]
        if ($Actual -ne $Entry.Value) { throw "Protected binary changed: $($Entry.Key): $Actual" }
    }
}

$Repo = Split-Path -Parent $PSScriptRoot
$Server = Join-Path $Repo "zig-out/bin/tensorfold-qualcomm-openai"
$Engine4B = Join-Path $Repo "zig-out/bin/tensorfold-qualcomm-exl3-forward-4b"
$Engine2B = Join-Path $Repo "zig-out/bin/tensorfold-qualcomm-exl3-forward"
$Launcher = Join-Path $PSScriptRoot "tensorfold_phone.sh"
$Config4B = "D:/op13-root/models/Qwen3.5-4B-exl3-4.00bpw/config.json"
$Config2B = "D:/op13-root/models/Qwen3.5-2B-exl3-8.00bpw/config.json"
foreach ($Path in @($Server, $Engine4B, $Engine2B, $Launcher, $Config4B, $Config2B)) {
    if (-not (Test-Path $Path)) { throw "Missing install input: $Path" }
}

Assert-Protected
Invoke-Adb shell "su -c 'if [ -x $Root/tensorfold ]; then $Root/tensorfold stop >/dev/null 2>&1 || true; fi'"
Invoke-Adb push $Server "$Root/tensorfold-qualcomm-openai"
Invoke-Adb push $Engine4B "$Root/tensorfold-qualcomm-exl3-forward-4b"
Invoke-Adb push $Engine2B "$Root/tensorfold-qualcomm-exl3-forward-2b"
Invoke-Adb push $Launcher "$Root/tensorfold"
Invoke-Adb push $Config4B "$ModelDir/config.json"
Invoke-Adb push $Config2B "/data/local/tmp/tensorfold-models/Qwen3.5-2B-exl3-8.00bpw/config.json"
Invoke-Adb shell "su -c 'chmod 755 $Root/tensorfold-qualcomm-openai $Root/tensorfold-qualcomm-exl3-forward-4b $Root/tensorfold-qualcomm-exl3-forward-2b $Root/tensorfold'"
Invoke-Adb shell "su -c '$Root/tensorfold doctor'"
Invoke-Adb shell "su -c '$Root/tensorfold shortcut'"
Assert-Protected

if ($Start) {
    Invoke-Adb shell "su -c '$Root/tensorfold start-ui'"
} else {
    Invoke-Adb shell "su -c '$Root/tensorfold stop'"
}
Write-Host "TensorFold Phone B install complete. Launcher: $Root/tensorfold start-ui"
