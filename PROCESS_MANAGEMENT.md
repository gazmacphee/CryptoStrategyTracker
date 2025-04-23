# Process Management System for Crypto Trading Platform

This document describes the centralized process management system for controlling all data processes in the cryptocurrency trading platform.

## Overview

The platform runs multiple background processes including:
- Data backfill processes
- Trading signal generation
- Machine learning training and prediction
- News retrieval
- Sentiment analysis
- Economic indicator updates

The process management system provides a central way to start, stop, and monitor all these processes, ensuring they run efficiently and can be properly shut down when needed.

## Quick Start

### Starting All Processes
To start all background processes:
```bash
python manage_processes.py start
```

This will start the process manager in monitor mode, which will:
1. Start all defined processes in priority order
2. Monitor their health and restart if they fail
3. Schedule processes to run according to their defined schedules

### Stopping All Processes
To gracefully stop all running processes:
```bash
python manage_processes.py stop
```

### Checking Process Status
To check the status of all processes:
```bash
python manage_processes.py status
```

### Manually Running Backfill
To manually trigger a data backfill:
```bash
python manage_processes.py backfill
```

## Advanced Usage

For more advanced control, you can use the process manager directly:

```bash
# Start all processes
python process_manager.py start

# Stop all processes
python process_manager.py stop

# Restart a specific process
python process_manager.py restart --process backfill

# Run the process monitor
python process_manager.py monitor
```

## Process Configuration

Processes are defined in the `MANAGED_PROCESSES` dictionary in `process_manager.py`. Each process configuration includes:

- `name`: Display name for the process
- `command`: Shell command to run
- `priority`: Order in which to start (lower numbers start first)
- `restart_on_failure`: Whether to automatically restart if the process fails
- `dependencies`: List of other processes that must be running first
- `max_restarts`: Maximum number of restart attempts
- `cooldown_seconds`: Cooldown period between restart attempts
- `schedule`: Optional scheduling information for periodic processes

## Adding New Processes

To add a new process to the system:

1. Edit `process_manager.py`
2. Add a new entry to the `MANAGED_PROCESSES` dictionary with appropriate configuration
3. Restart the process manager

## Logs and Monitoring

All processes write to their own log files:
- Process manager logs: `process_manager.log`
- Process management wrapper logs: `process_management.log`
- Individual process logs: `{process_id}.log`

## Process Flow

1. The `manage_processes.py` script provides a simple interface for users
2. It controls the `process_manager.py` script, which manages individual processes
3. Each process runs independently, with dependencies managed by the process manager
4. The process manager monitors health and handles restarts as needed

## Architecture

```
┌─ manage_processes.py (User Interface)
│
└─► process_manager.py (Process Controller)
    │
    ├─► backfill_database.py (Data Gathering)
    │
    ├─► generate_signals.py (Signal Generation)
    │
    ├─► advanced_ml.py (ML Training & Prediction)
    │
    ├─► crypto_news.py (News Retrieval)
    │
    ├─► sentiment_scraper.py (Sentiment Analysis)
    │
    └─► economic_indicators.py (Economic Data Updates)
```

## Troubleshooting

If processes are not starting or stopping correctly:

1. Check the log files for errors
2. Ensure no lock files are present (`.process_manager.lock`, `.backfill_lock`)
3. Verify that all required Python modules are installed
4. Check for any zombie processes and manually kill if necessary:
   ```bash
   ps aux | grep python
   kill <PID>
   ```

## Windows Compatibility

For Windows environments, the process management system works with some limitations:
- Process signal handling is different
- Some process status checks may not work as expected
- Use `python` instead of `./` to run scripts