# Log a serial port to a file (one line per received line, prefixed with the laptop's local time).
#   powershell -ExecutionPolicy Bypass -File scripts\serial_log.ps1 -Port COM3 -Out logs\board_energy.log -Seconds 600
param(
    [string]$Port = "COM3",
    [int]$Baud = 115200,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Seconds = 600
)
$sp = New-Object System.IO.Ports.SerialPort $Port, $Baud, "None", 8, "One"
$sp.ReadTimeout = 500
$sp.NewLine = "`n"
$sp.Open()
$end = (Get-Date).AddSeconds($Seconds)
try {
    while ((Get-Date) -lt $end) {
        try {
            $line = $sp.ReadLine().TrimEnd("`r")
            $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
            Add-Content -Path $Out -Value "$stamp  $line" -Encoding utf8
        } catch [System.TimeoutException] { }
    }
} finally {
    $sp.Close()
}
