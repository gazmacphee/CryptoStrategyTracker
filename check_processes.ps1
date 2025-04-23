# CryptoStrategyTracker Process Status Check
# This script provides a user-friendly way to view process status

# Define colors for better readability
$Colors = @{
    "Title" = "Cyan"
    "Running" = "Green"
    "Stopped" = "Red"
    "Warning" = "Yellow"
    "Info" = "White"
}

function Show-Header {
    Write-Host "`n==============================================" -ForegroundColor $Colors.Title
    Write-Host "  CryptoStrategyTracker Process Status Check" -ForegroundColor $Colors.Title
    Write-Host "==============================================" -ForegroundColor $Colors.Title
    Write-Host
}

function Get-ProcessInfo {
    try {
        $processInfoJson = Get-Content .process_info.json -ErrorAction Stop | ConvertFrom-Json
        return $processInfoJson
    }
    catch {
        Write-Host "Could not read process info file. Is the application initialized?" -ForegroundColor $Colors.Warning
        return $null
    }
}

function Check-ProcessRunning($pid) {
    if ($null -eq $pid -or $pid -eq 0) {
        return $false
    }
    
    try {
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        return $null -ne $process
    }
    catch {
        return $false
    }
}

function Show-ProcessStatus {
    $processInfo = Get-ProcessInfo
    if ($null -eq $processInfo) {
        # Even if we don't have process info, try to detect known processes
        $pythonProcesses = Get-Process -Name python -ErrorAction SilentlyContinue
        if ($pythonProcesses) {
            Write-Host "DETECTED PYTHON PROCESSES:" -ForegroundColor $Colors.Title
            Write-Host "-------------------------" -ForegroundColor $Colors.Title
            
            # Try to identify common app processes by their command line
            $knownProcesses = @()
            foreach ($proc in $pythonProcesses) {
                try {
                    $cmdLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
                    
                    # Match known application scripts
                    if ($cmdLine -match "process_manager\.py") {
                        $knownProcesses += [PSCustomObject]@{
                            Name = "Process Manager"
                            PID = $proc.Id
                            Command = $cmdLine
                        }
                    }
                    elseif ($cmdLine -match "app\.py") {
                        $knownProcesses += [PSCustomObject]@{
                            Name = "Streamlit App"
                            PID = $proc.Id
                            Command = $cmdLine
                        }
                    }
                    elseif ($cmdLine -match "backfill|download") {
                        $knownProcesses += [PSCustomObject]@{
                            Name = "Data Backfill"
                            PID = $proc.Id
                            Command = $cmdLine
                        }
                    }
                } catch {
                    # Skip if we can't get command line
                }
            }
            
            # Display known processes
            if ($knownProcesses.Count -gt 0) {
                foreach ($proc in $knownProcesses) {
                    Write-Host "$($proc.Name): " -NoNewline
                    Write-Host "RUNNING" -ForegroundColor $Colors.Running -NoNewline
                    Write-Host " (PID: $($proc.PID))"
                    Write-Host "  Command: $($proc.Command)" -ForegroundColor $Colors.Info
                    Write-Host
                }
            } else {
                Write-Host "No recognized application processes detected." -ForegroundColor $Colors.Warning
                Write-Host "Some Python processes are running, but they may not belong to the application." -ForegroundColor $Colors.Warning
                Write-Host
            }
        }
        
        return
    }
    
    # If we have process info, display it as normal
    Write-Host "MANAGED PROCESSES:" -ForegroundColor $Colors.Title
    Write-Host "----------------" -ForegroundColor $Colors.Title
    
    if ($processInfo.processes.PSObject.Properties.Count -eq 0) {
        Write-Host "No managed processes found in process info file." -ForegroundColor $Colors.Warning
        Write-Host
    } else {
        foreach ($process in $processInfo.processes) {
            # Fix the function call syntax to avoid variable overwriting issues
            $processId = $process.pid
            $isRunning = Check-ProcessRunning -pid $processId
            $status = if ($isRunning) { "RUNNING" } else { "STOPPED" }
            $color = if ($isRunning) { $Colors.Running } else { $Colors.Stopped }
            
            Write-Host "$($process.name): " -NoNewline
            Write-Host $status -ForegroundColor $color -NoNewline
            Write-Host " (PID: $($process.pid))"
            Write-Host "  Command: $($process.command)" -ForegroundColor $Colors.Info
            Write-Host "  Last Status: $($process.status)" -ForegroundColor $Colors.Info
            Write-Host "  Started: $($process.start_time)" -ForegroundColor $Colors.Info
            Write-Host
        }
    }
    
    # Show scheduled processes
    if ($processInfo.scheduled_processes -and $processInfo.scheduled_processes.Count -gt 0) {
        Write-Host "SCHEDULED PROCESSES:" -ForegroundColor $Colors.Title
        Write-Host "-------------------" -ForegroundColor $Colors.Title
        
        foreach ($process in $processInfo.scheduled_processes) {
            Write-Host "$($process.name): " -NoNewline
            Write-Host "Scheduled" -ForegroundColor $Colors.Info
            Write-Host "  Command: $($process.command)" -ForegroundColor $Colors.Info
            Write-Host "  Schedule: $($process.schedule)" -ForegroundColor $Colors.Info
            Write-Host "  Last Run: $($process.last_run)" -ForegroundColor $Colors.Info
            Write-Host
        }
    }
    
    # Show last run times for background processes
    if ($processInfo.last_run_times -and $processInfo.last_run_times.PSObject.Properties.Count -gt 0) {
        Write-Host "BACKGROUND PROCESSES (Last Run):" -ForegroundColor $Colors.Title
        Write-Host "------------------------------" -ForegroundColor $Colors.Title
        
        $processInfo.last_run_times.PSObject.Properties | ForEach-Object {
            Write-Host "$($_.Name): " -NoNewline
            Write-Host "$($_.Value)" -ForegroundColor $Colors.Info
        }
        Write-Host
    }
}

function Show-SystemInfo {
    Write-Host "SYSTEM INFO:" -ForegroundColor $Colors.Title
    Write-Host "-----------" -ForegroundColor $Colors.Title
    
    # Get overall Python processes
    $pythonProcesses = Get-Process -Name python -ErrorAction SilentlyContinue
    $pythonCount = if ($pythonProcesses) { $pythonProcesses.Count } else { 0 }
    
    Write-Host "Total Python Processes: $pythonCount" -ForegroundColor $Colors.Info
    
    # Check if process manager is running
    $processManagerRunning = $false
    if ($pythonProcesses) {
        foreach ($proc in $pythonProcesses) {
            try {
                $cmdLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
                if ($cmdLine -and $cmdLine -match "process_manager\.py") {
                    $processManagerRunning = $true
                    Write-Host "Process Manager: " -NoNewline
                    Write-Host "RUNNING" -ForegroundColor $Colors.Running -NoNewline
                    Write-Host " (PID: $($proc.Id))"
                    break
                }
            } catch {
                # Unable to retrieve command line, continue checking other processes
            }
        }
    }
    
    # Check for process manager in another way if not found
    if (-not $processManagerRunning) {
        # Check if there's a .process_manager.lock file (which indicates running)
        if (Test-Path ".process_manager.lock") {
            $lockFileAge = (Get-Date) - (Get-Item ".process_manager.lock").LastWriteTime
            if ($lockFileAge.TotalMinutes -lt 5) {
                $processManagerRunning = $true
                Write-Host "Process Manager: " -NoNewline
                Write-Host "RUNNING" -ForegroundColor $Colors.Running -NoNewline
                Write-Host " (lock file present)"
            }
        }
    }
    
    if (-not $processManagerRunning) {
        Write-Host "Process Manager: " -NoNewline
        Write-Host "STOPPED" -ForegroundColor $Colors.Stopped
    }
    
    # Latest log entries
    Write-Host "`nLATEST LOG ENTRIES:" -ForegroundColor $Colors.Title
    Write-Host "-----------------" -ForegroundColor $Colors.Title
    
    if (Test-Path "process_manager.log") {
        Write-Host "Process Manager Log:" -ForegroundColor $Colors.Info
        $logEntries = Get-Content "process_manager.log" -Tail 5
        foreach ($entry in $logEntries) {
            if ($entry -match "ERROR|error|failed|FAILED") {
                Write-Host "  $entry" -ForegroundColor $Colors.Stopped
            }
            elseif ($entry -match "WARN|warning|WARNING") {
                Write-Host "  $entry" -ForegroundColor $Colors.Warning
            }
            else {
                Write-Host "  $entry" -ForegroundColor $Colors.Info
            }
        }
    }
    else {
        Write-Host "No process manager log found" -ForegroundColor $Colors.Warning
    }
    
    Write-Host
}

function Show-CommandHelp {
    Write-Host "`nCOMMAND REFERENCE:" -ForegroundColor $Colors.Title
    Write-Host "-----------------" -ForegroundColor $Colors.Title
    Write-Host "Start all processes:    " -NoNewline -ForegroundColor $Colors.Info
    Write-Host "python process_manager.py start"
    Write-Host "Start and keep running: " -NoNewline -ForegroundColor $Colors.Info
    Write-Host "python process_manager.py run"
    Write-Host "Stop all processes:     " -NoNewline -ForegroundColor $Colors.Info
    Write-Host "python process_manager.py stop"
    Write-Host "Restart all processes:  " -NoNewline -ForegroundColor $Colors.Info
    Write-Host "python process_manager.py restart"
    Write-Host "Start a specific process:" -NoNewline -ForegroundColor $Colors.Info
    Write-Host "python process_manager.py start --process backfill"
    Write-Host "Monitor all processes:  " -NoNewline -ForegroundColor $Colors.Info
    Write-Host "python process_manager.py monitor"
    Write-Host "`nTo check status again: " -NoNewline -ForegroundColor $Colors.Info
    Write-Host ".\check_processes.ps1"
    Write-Host
}

# Main execution flow
Show-Header
Show-ProcessStatus
Show-SystemInfo
Show-CommandHelp