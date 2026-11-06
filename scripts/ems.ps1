param([switch]$Stop, [switch]$Public)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectRoot 'runtime'
$statePath = Join-Path $runtimePath 'wsl-keeper.json'
$keeper = $null

if (Test-Path -LiteralPath $statePath) {
    $saved = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $candidate = Get-Process -Id $saved.Id -ErrorAction SilentlyContinue
    if ($candidate -and $candidate.ProcessName -eq 'wsl' -and
        $candidate.StartTime.ToUniversalTime().Ticks.ToString() -eq $saved.StartTicks) {
        $keeper = $candidate
    }
}

if ($Stop) {
    & wsl.exe -d Ubuntu -u root --cd $projectRoot --exec docker compose --profile public down
    if ($LASTEXITCODE -ne 0) { throw 'Could not stop EMS containers.' }
    if ($keeper) { Stop-Process -Id $keeper.Id }
    if (Test-Path -LiteralPath $statePath) { Remove-Item -LiteralPath $statePath }
    Write-Host 'EMS stopped. Database volume preserved.'
    return
}

New-Item -ItemType Directory -Force -Path $runtimePath | Out-Null
if (-not $keeper) {
    # systemd services alone do not keep WSL alive after its foreground commands exit.
    $keeper = Start-Process -FilePath "$env:SystemRoot\System32\wsl.exe" `
        -ArgumentList @('-d', 'Ubuntu', '--exec', 'sleep', 'infinity') `
        -WindowStyle Hidden -PassThru
    @{
        Id = $keeper.Id
        StartTicks = $keeper.StartTime.ToUniversalTime().Ticks.ToString()
    } | ConvertTo-Json | Set-Content -LiteralPath $statePath
}

& wsl.exe -d Ubuntu -u root --exec systemctl start docker
if ($LASTEXITCODE -ne 0) { throw 'Could not start Docker in Ubuntu.' }
& wsl.exe -d Ubuntu -u root --cd $projectRoot --exec docker compose build api
if ($LASTEXITCODE -ne 0) { throw 'EMS build failed.' }
$profileArgs = @()
if ($Public) { $profileArgs = @('--profile', 'public') }
& wsl.exe -d Ubuntu -u root --cd $projectRoot --exec docker compose @profileArgs up -d --wait --wait-timeout 120
if ($LASTEXITCODE -ne 0) { throw 'EMS startup failed. Check docker compose logs.' }
Write-Host 'Local URL: http://localhost:8501'
Write-Host 'Grafana: http://localhost:3000/d/ems-dispatch/ems-dispatch'
if (-not $Public) { return }
$tunnelLog = & wsl.exe -d Ubuntu -u root --cd $projectRoot --exec docker compose logs --tail=80 tunnel
if ($LASTEXITCODE -ne 0) { throw 'Could not read tunnel logs.' }
$publicUrls = [regex]::Matches(($tunnelLog -join "`n"), 'https://[a-z0-9-]+\.trycloudflare\.com')
if ($publicUrls.Count -gt 0) {
    $publicUrl = $publicUrls[$publicUrls.Count - 1].Value
    Set-Content -LiteralPath (Join-Path $runtimePath 'public-url.txt') -Value $publicUrl
    Write-Host "Public URL: $publicUrl"
} else {
    Write-Host 'Tunnel is still connecting. Read its logs with docker compose logs tunnel in Ubuntu.'
}
