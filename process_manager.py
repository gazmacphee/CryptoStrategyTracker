"""
Process Manager for Cryptocurrency Trading Analysis Platform

This module provides centralized control over all running processes including:
- Data gathering/backfill processes
- Machine learning model training and prediction
- Technical indicator calculations
- Trading signal generation
- News retrieval
- Sentiment analysis

Usage:
    To start all processes: python process_manager.py start
    To stop all processes: python process_manager.py stop
    To check status: python process_manager.py status
"""

import os
import sys
import time
import signal
import logging
import subprocess
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Import psutil with fallback for cross-platform compatibility
try:
    import psutil
    PSUTIL_AVAILABLE = True
except (ImportError, OSError):
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available or not working properly on this system. "
                   "Process management will use basic OS functions instead.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_manager.log"),
        logging.StreamHandler()
    ]
)

# Constants
PROCESS_LOCK_FILE = ".process_manager.lock"
PROCESS_INFO_FILE = ".process_info.json"
BACKFILL_LOCK_FILE = ".backfill_lock"  # Existing lock file used by the backfill process

# Process configuration - defines all managed processes
MANAGED_PROCESSES = {
    "backfill": {
        "name": "Data Backfill",
        "command": ["python", "backfill_database.py", "--continuous", "--interval", "15"],
        "priority": 1,  # Lower numbers start first
        "restart_on_failure": True,
        "dependencies": [],  # No dependencies required
        "max_restarts": 3,
        "cooldown_seconds": 60,
    },
    "ml_training": {
        "name": "ML Model Training",
        "command": ["python", "advanced_ml.py", "--train"],
        "priority": 3,
        "restart_on_failure": True,
        "dependencies": ["backfill"],  # Requires backfill to run first
        "max_restarts": 2,
        "cooldown_seconds": 300,  # 5 minutes between restarts
        "schedule": {
            "interval_hours": 24,  # Run daily
            "at_startup": True,
        }
    },
    "signal_generator": {
        "name": "Trading Signal Generator",
        "command": ["python", "generate_signals.py", "--continuous"],
        "priority": 2,
        "restart_on_failure": True,
        "dependencies": ["backfill"],
        "max_restarts": 3,
        "cooldown_seconds": 120,
    },
    "news_retrieval": {
        "name": "News Retrieval",
        "command": ["python", "crypto_news.py", "--fetch"],
        "priority": 4,
        "restart_on_failure": True,
        "dependencies": [],
        "max_restarts": 3,
        "cooldown_seconds": 300,
        "schedule": {
            "interval_hours": 1,  # Run hourly
            "at_startup": True,
        }
    },
    "sentiment_analysis": {
        "name": "Sentiment Analysis",
        "command": ["python", "sentiment_scraper.py"],
        "priority": 5,
        "restart_on_failure": True,
        "dependencies": ["news_retrieval"],
        "max_restarts": 3,
        "cooldown_seconds": 300,
        "schedule": {
            "interval_hours": 4,  # Run every 4 hours
            "at_startup": True,
        }
    },
    "economic_indicators": {
        "name": "Economic Indicators Update",
        "command": ["python", "economic_indicators.py", "--update"],
        "priority": 6,
        "restart_on_failure": True,
        "dependencies": [],
        "max_restarts": 2,
        "cooldown_seconds": 600,
        "schedule": {
            "interval_hours": 24,  # Run daily
            "at_startup": True,
        }
    }
}


class ProcessManager:
    """Manages all processes for the cryptocurrency trading analysis platform."""

    def __init__(self):
        self.processes: Dict[str, Dict] = {}
        self.last_run_times: Dict[str, datetime] = {}
        self._load_process_state()
        self.shutdown_requested = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals to gracefully shut down processes."""
        logging.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.shutdown_requested = True
        self.stop_all_processes()
        sys.exit(0)

    def _load_process_state(self) -> None:
        """Load process state from the info file if it exists."""
        if os.path.exists(PROCESS_INFO_FILE):
            try:
                with open(PROCESS_INFO_FILE, 'r') as f:
                    info = json.load(f)
                    
                # Convert string dates back to datetime objects
                for proc_id, last_run in info.get('last_run_times', {}).items():
                    if last_run:
                        self.last_run_times[proc_id] = datetime.fromisoformat(last_run)
                
                logging.info(f"Loaded process state from {PROCESS_INFO_FILE}")
            except Exception as e:
                logging.error(f"Error loading process state: {e}")

    def _save_process_state(self) -> None:
        """Save the current process state to the info file."""
        info = {
            'last_update': datetime.now().isoformat(),
            'processes': {
                proc_id: {
                    'pid': proc_info.get('pid'),
                    'status': proc_info.get('status'),
                    'start_time': proc_info.get('start_time'),
                    'restarts': proc_info.get('restarts', 0)
                } for proc_id, proc_info in self.processes.items()
            },
            'last_run_times': {
                proc_id: dt.isoformat() if dt else None 
                for proc_id, dt in self.last_run_times.items()
            }
        }
        
        try:
            with open(PROCESS_INFO_FILE, 'w') as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving process state: {e}")

    def create_lock_file(self) -> bool:
        """Create a lock file to indicate the process manager is running."""
        if os.path.exists(PROCESS_LOCK_FILE):
            # Check if the process is still running
            try:
                with open(PROCESS_LOCK_FILE, 'r') as f:
                    pid = int(f.read().strip())
                
                if self._is_process_running(pid) and pid != os.getpid():
                    logging.warning(f"Process manager already running with PID {pid}")
                    return False
            except (ValueError, FileNotFoundError):
                pass
        
        # Create or update the lock file with current PID
        with open(PROCESS_LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        return True

    def remove_lock_file(self) -> None:
        """Remove the lock file when the process manager stops."""
        if os.path.exists(PROCESS_LOCK_FILE):
            try:
                os.remove(PROCESS_LOCK_FILE)
                logging.info("Removed process manager lock file")
            except Exception as e:
                logging.error(f"Error removing lock file: {e}")

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """Check if a process with the given PID is running."""
        if not pid:
            return False
            
        # Use psutil if available
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(pid)
                return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, Exception):
                return False
        else:
            # Fallback to basic OS check
            try:
                # On UNIX-like systems, sending signal 0 checks if process exists
                os.kill(pid, 0)
                return True
            except (OSError, ProcessLookupError):
                return False

    def start_all_processes(self) -> None:
        """Start all managed processes in priority order."""
        if not self.create_lock_file():
            logging.error("Could not create lock file - another instance may be running")
            return
        
        logging.info("Starting all processes...")
        
        # Sort processes by priority
        sorted_processes = sorted(MANAGED_PROCESSES.items(), 
                                 key=lambda x: x[1].get('priority', 999))
        
        for proc_id, config in sorted_processes:
            self._start_process(proc_id, config)
            time.sleep(2)  # Small delay between process starts to prevent resource contention
        
        self._save_process_state()
        logging.info("All processes started")

    def _should_run_scheduled_process(self, proc_id: str, config: Dict) -> bool:
        """Determine if a scheduled process should run now."""
        schedule = config.get('schedule')
        if not schedule:
            return True  # No schedule means run whenever called
        
        # Get the last run time or set to None if never run
        last_run = self.last_run_times.get(proc_id)
        
        # Run at startup if configured and never run before
        if schedule.get('at_startup', False) and not last_run:
            return True
        
        # Check if enough time has passed since last run
        if last_run:
            interval_hours = schedule.get('interval_hours', 24)
            next_run = last_run + timedelta(hours=interval_hours)
            return datetime.now() >= next_run
        
        return True  # No record of last run, so run now

    def _start_process(self, proc_id: str, config: Dict) -> None:
        """Start a specific process if all dependencies are met."""
        # Check if process is already running
        if proc_id in self.processes and self.processes[proc_id].get('pid'):
            pid = self.processes[proc_id]['pid']
            if self._is_process_running(pid):
                logging.info(f"Process {config['name']} already running with PID {pid}")
                return
        
        # Check dependencies
        for dep in config.get('dependencies', []):
            # Special case for backfill, which might be running outside the process manager
            if dep == "backfill":
                # Check if a backfill process is running using ps command
                is_backfill_running = False
                try:
                    ps_output = subprocess.run(
                        ["ps", "aux"], 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    ).stdout
                    
                    for line in ps_output.split('\n'):
                        if "backfill_database.py" in line and "python" in line:
                            is_backfill_running = True
                            break
                except Exception:
                    # If we can't check, assume it's not running
                    is_backfill_running = False
                
                # Also check for the backfill lock file as an indicator
                if os.path.exists(BACKFILL_LOCK_FILE):
                    is_backfill_running = True
                    
                if is_backfill_running:
                    # Backfill is running outside our process management
                    continue
            
            # Standard dependency check
            if dep not in self.processes or not self.processes[dep].get('pid'):
                logging.warning(f"Cannot start {config['name']} - dependency {dep} not running")
                return
            
            # Check if dependency is still running
            dep_pid = self.processes[dep].get('pid')
            if not self._is_process_running(dep_pid):
                logging.warning(f"Cannot start {config['name']} - dependency {dep} (PID {dep_pid}) not running")
                return
        
        # Check schedule
        if not self._should_run_scheduled_process(proc_id, config):
            logging.info(f"Skipping scheduled process {config['name']} - not time to run yet")
            return
        
        try:
            logging.info(f"Starting process: {config['name']} ({' '.join(config['command'])})")
            
            # Start the process and redirect output to logs
            log_file = open(f"{proc_id}.log", "a")
            process = subprocess.Popen(
                config['command'],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Record process information
            self.processes[proc_id] = {
                'pid': process.pid,
                'status': 'running',
                'start_time': datetime.now().isoformat(),
                'restarts': self.processes.get(proc_id, {}).get('restarts', 0),
                'log_file': log_file
            }
            
            # Record last run time for scheduled processes
            self.last_run_times[proc_id] = datetime.now()
            
            logging.info(f"Started process {config['name']} with PID {process.pid}")
            
        except Exception as e:
            logging.error(f"Error starting process {config['name']}: {e}")
            self.processes[proc_id] = {
                'pid': None,
                'status': 'failed',
                'error': str(e),
                'restarts': self.processes.get(proc_id, {}).get('restarts', 0)
            }

    def stop_process(self, proc_id: str) -> bool:
        """Stop a specific process by its ID."""
        if proc_id not in self.processes or not self.processes[proc_id].get('pid'):
            logging.warning(f"Process {proc_id} not found or not running")
            return False
        
        pid = self.processes[proc_id]['pid']
        
        # Close the log file if it exists
        log_file = self.processes[proc_id].get('log_file')
        if log_file and not log_file.closed:
            log_file.close()
        
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(pid)
                logging.info(f"Stopping process {proc_id} (PID {pid})...")
                
                # Try graceful termination first
                process.terminate()
                
                # Wait for up to 5 seconds for the process to terminate
                gone, alive = psutil.wait_procs([process], timeout=5)
                
                # If still alive, kill it
                if alive:
                    logging.warning(f"Process {proc_id} (PID {pid}) did not terminate gracefully, killing...")
                    process.kill()
                
                # Update process status
                self.processes[proc_id]['status'] = 'stopped'
                self.processes[proc_id]['pid'] = None
                
                logging.info(f"Process {proc_id} stopped")
                return True
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError) as e:
                logging.warning(f"Process {proc_id} (PID {pid}) already terminated: {e}")
                self.processes[proc_id]['status'] = 'stopped'
                self.processes[proc_id]['pid'] = None
                return True
            except Exception as e:
                logging.error(f"Error stopping process {proc_id} (PID {pid}) with psutil: {e}")
                # Fall back to basic OS approach
        
        # Basic OS process termination (used when psutil is not available)
        try:
            logging.info(f"Stopping process {proc_id} (PID {pid}) using OS signals...")
            # Try sending SIGTERM signal first for graceful shutdown
            os.kill(pid, signal.SIGTERM)
            
            # Give the process a moment to terminate
            for _ in range(10):  # Wait up to 5 seconds
                time.sleep(0.5)
                try:
                    # Check if process still exists
                    os.kill(pid, 0)
                except OSError:
                    # Process no longer exists
                    break
            else:
                # Process still exists after waiting, try SIGKILL
                try:
                    logging.warning(f"Process {proc_id} (PID {pid}) did not terminate with SIGTERM, sending SIGKILL...")
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass  # Process likely already terminated
            
            # Update process status
            self.processes[proc_id]['status'] = 'stopped'
            self.processes[proc_id]['pid'] = None
            logging.info(f"Process {proc_id} stopped")
            return True
            
        except (OSError, ProcessLookupError) as e:
            logging.warning(f"Process {proc_id} (PID {pid}) already terminated or not accessible: {e}")
            self.processes[proc_id]['status'] = 'stopped'
            self.processes[proc_id]['pid'] = None
            return True
        except Exception as e:
            logging.error(f"Error stopping process {proc_id} (PID {pid}): {e}")
            return False

    def stop_all_processes(self) -> None:
        """Stop all running processes in reverse priority order."""
        logging.info("Stopping all processes...")
        
        # Sort processes by priority (high to low, so highest priority stops last)
        sorted_processes = sorted(self.processes.items(), 
                                 key=lambda x: MANAGED_PROCESSES.get(x[0], {}).get('priority', 0),
                                 reverse=True)
        
        for proc_id, _ in sorted_processes:
            self.stop_process(proc_id)
            time.sleep(1)  # Small delay between process stops
        
        self._save_process_state()
        self.remove_lock_file()
        logging.info("All processes stopped")

    def restart_process(self, proc_id: str) -> bool:
        """Restart a specific process by its ID."""
        if proc_id not in MANAGED_PROCESSES:
            logging.error(f"Unknown process ID: {proc_id}")
            return False
        
        # Stop the process if it's running
        if proc_id in self.processes and self.processes[proc_id].get('pid'):
            self.stop_process(proc_id)
        
        # Create a copy of the configuration
        config = MANAGED_PROCESSES[proc_id].copy()
        
        # Temporarily remove any schedule to force immediate execution
        if 'schedule' in config:
            temp_schedule = config.pop('schedule')
            logging.info(f"Temporarily removing schedule to force immediate execution of {config['name']}")
            
            # Force update of last run time to prevent cooldown issues
            self.last_run_times[proc_id] = datetime.now() - timedelta(days=1)
        
        # Start the process again
        self._start_process(proc_id, config)
        self._save_process_state()
        
        return True

    def check_processes(self) -> None:
        """Check the status of all processes and restart if needed."""
        for proc_id, config in MANAGED_PROCESSES.items():
            # Skip if not monitoring this process yet
            if proc_id not in self.processes:
                continue
            
            proc_info = self.processes[proc_id]
            pid = proc_info.get('pid')
            
            # Skip if no PID (not started or already stopped)
            if not pid:
                continue
            
            # Check if process is still running
            if not self._is_process_running(pid):
                logging.warning(f"Process {config['name']} (PID {pid}) is not running")
                
                # Update status
                proc_info['status'] = 'stopped'
                proc_info['pid'] = None
                
                # Restart if configured to do so
                if config.get('restart_on_failure', False):
                    max_restarts = config.get('max_restarts', 3)
                    restarts = proc_info.get('restarts', 0)
                    
                    if restarts < max_restarts:
                        # Check cooldown period
                        cooldown_seconds = config.get('cooldown_seconds', 60)
                        last_start_time = datetime.fromisoformat(proc_info.get('start_time', '2000-01-01T00:00:00'))
                        
                        if (datetime.now() - last_start_time).total_seconds() > cooldown_seconds:
                            logging.info(f"Restarting process {config['name']} (restart {restarts + 1}/{max_restarts})")
                            proc_info['restarts'] = restarts + 1
                            self._start_process(proc_id, config)
                        else:
                            logging.info(f"Waiting for cooldown before restarting {config['name']}")
                    else:
                        logging.warning(f"Process {config['name']} reached maximum restarts ({max_restarts})")
        
        self._save_process_state()

    def get_process_status(self) -> Dict:
        """Get the status of all processes."""
        status = {}
        
        for proc_id, config in MANAGED_PROCESSES.items():
            proc_info = self.processes.get(proc_id, {})
            pid = proc_info.get('pid')
            
            # Check if the process is actually running
            is_running = False
            external_pid = None
            
            # First check our managed process
            if pid:
                is_running = self._is_process_running(pid)
                
                # Update internal state if our record is wrong
                if not is_running and proc_info.get('status') == 'running':
                    proc_info['status'] = 'stopped'
                    proc_info['pid'] = None
            
            # Special case for backfill process which might be running outside our control
            if proc_id == "backfill" and not is_running:
                # Check if a backfill process is running using ps command
                try:
                    ps_output = subprocess.run(
                        ["ps", "aux"], 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    ).stdout
                    
                    for line in ps_output.split('\n'):
                        if "backfill_database.py" in line and "python" in line:
                            parts = line.split()
                            if len(parts) > 1:
                                try:
                                    external_pid = int(parts[1])
                                    is_running = True
                                    break
                                except ValueError:
                                    pass
                except Exception:
                    # If we can't check, assume it's not running
                    pass
                
                # Also check for the backfill lock file as an indicator
                if os.path.exists(BACKFILL_LOCK_FILE):
                    is_running = True
            
            # Get the last run time
            last_run = self.last_run_times.get(proc_id)
            last_run_str = last_run.isoformat() if last_run else "Never"
            
            # Next scheduled run for scheduled processes
            next_run_str = "N/A"
            if last_run and config.get('schedule'):
                interval_hours = config.get('schedule', {}).get('interval_hours', 24)
                next_run = last_run + timedelta(hours=interval_hours)
                next_run_str = next_run.isoformat()
            
            status[proc_id] = {
                'name': config['name'],
                'status': 'running' if is_running else proc_info.get('status', 'not started'),
                'pid': external_pid or (pid if is_running else None),
                'start_time': proc_info.get('start_time'),
                'restarts': proc_info.get('restarts', 0),
                'last_run': last_run_str,
                'next_run': next_run_str,
                'command': ' '.join(config['command'])
            }
        
        return status

    def print_status(self) -> None:
        """Print the status of all processes to the console."""
        status = self.get_process_status()
        
        print("\n===== Process Manager Status =====")
        print(f"Time: {datetime.now().isoformat()}")
        print("=================================")
        
        for proc_id, info in status.items():
            status_str = info['status'].upper()
            if status_str == 'RUNNING':
                status_str = f"\033[92m{status_str}\033[0m"  # Green for running
            elif status_str == 'STOPPED':
                status_str = f"\033[91m{status_str}\033[0m"  # Red for stopped
            elif status_str == 'FAILED':
                status_str = f"\033[91m{status_str}\033[0m"  # Red for failed
            
            print(f"\nProcess: {info['name']} ({proc_id})")
            print(f"Status: {status_str}")
            print(f"PID: {info['pid'] or 'N/A'}")
            print(f"Command: {info['command']}")
            print(f"Start Time: {info['start_time'] or 'N/A'}")
            print(f"Restarts: {info['restarts']}")
            print(f"Last Run: {info['last_run']}")
            print(f"Next Run: {info['next_run']}")
        
        print("\n=================================")

    def run_monitor(self) -> None:
        """Run the process monitor loop."""
        if not self.create_lock_file():
            logging.error("Could not create lock file - another instance may be running")
            return
        
        logging.info("Process monitor started")
        
        try:
            while not self.shutdown_requested:
                self.check_processes()
                
                # Start any scheduled processes that should run now
                for proc_id, config in MANAGED_PROCESSES.items():
                    if config.get('schedule') and self._should_run_scheduled_process(proc_id, config):
                        logging.info(f"Running scheduled process {config['name']}")
                        self._start_process(proc_id, config)
                
                # Sleep for a bit before checking again
                time.sleep(30)
                
        except KeyboardInterrupt:
            logging.info("Process monitor interrupted")
        finally:
            self._save_process_state()
            self.remove_lock_file()
            logging.info("Process monitor stopped")

    def cleanup(self) -> None:
        """Clean up resources and stop all processes before exiting."""
        self.stop_all_processes()
        self.remove_lock_file()


def create_generate_signals_script():
    """Create a script to generate trading signals if it doesn't exist."""
    script_path = "generate_signals.py"
    
    if os.path.exists(script_path):
        return
    
    script_content = """#!/usr/bin/env python
'''
Trading Signal Generator Script

This script analyzes historical price data and generates trading signals
continuously or on-demand.

Usage:
    python generate_signals.py [--continuous]
    
    --continuous: Run in continuous mode, generating signals as new data arrives
'''
"""
    script_content += """
import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("signals.log"),
        logging.StreamHandler()
    ]
)

def setup_argparse():
    \"\"\"Parse command-line arguments.\"\"\"
    parser = argparse.ArgumentParser(description='Generate trading signals')
    parser.add_argument('--continuous', action='store_true', 
                        help='Run in continuous mode, generating signals as new data arrives')
    return parser.parse_args()

def generate_signals():
    \"\"\"Generate trading signals from historical data.\"\"\"
    try:
        logging.info("Generating trading signals...")
        
        # Import modules inside function to handle import errors gracefully
        import pandas as pd
        from trading_signals import save_trading_signals
        from strategy import evaluate_buy_sell_signals
        from database import get_db_connection
        from binance_api import get_available_symbols
        from indicators import add_bollinger_bands, add_macd, add_rsi, add_ema
        
        # Get available trading pairs
        available_symbols = get_available_symbols()
        
        # Define intervals to analyze
        intervals = ['1h', '4h', '1d']
        
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to connect to database")
            return False
        
        cursor = conn.cursor()
        signals_generated = 0
        
        # Process each symbol and interval
        for symbol in available_symbols[:10]:  # Process top 10 symbols
            for interval in intervals:
                try:
                    logging.info(f"Processing {symbol}/{interval}")
                    
                    # Get historical data from database
                    query = \"\"\"
                    SELECT timestamp, open, high, low, close, volume
                    FROM historical_data
                    WHERE symbol = %s AND interval = %s
                    ORDER BY timestamp DESC
                    LIMIT 500
                    \"\"\"
                    
                    cursor.execute(query, (symbol, interval))
                    rows = cursor.fetchall()
                    
                    if not rows:
                        logging.warning(f"No data found for {symbol}/{interval}")
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df = df.sort_values('timestamp')
                    
                    # Add technical indicators
                    df = add_bollinger_bands(df)
                    df = add_macd(df)
                    df = add_rsi(df)
                    df = add_ema(df)
                    
                    # Apply strategy to generate signals
                    result_df = evaluate_buy_sell_signals(df)
                    
                    # Save signals to database
                    success = save_trading_signals(result_df, symbol, interval)
                    
                    if success:
                        signals = result_df[(result_df['buy_signal'] == True) | (result_df['sell_signal'] == True)]
                        signals_count = len(signals)
                        signals_generated += signals_count
                        logging.info(f"Generated {signals_count} signals for {symbol}/{interval}")
                    
                except Exception as e:
                    logging.error(f"Error generating signals for {symbol}/{interval}: {e}")
        
        if conn:
            conn.close()
        
        logging.info(f"Generated a total of {signals_generated} trading signals")
        return signals_generated > 0
    
    except ImportError as e:
        logging.error(f"Import error: {e}")
        return False
    except Exception as e:
        logging.error(f"Error generating signals: {e}")
        return False

def main():
    \"\"\"Main entry point for signal generation.\"\"\"
    args = setup_argparse()
    
    if args.continuous:
        logging.info("Starting continuous signal generation...")
        try:
            while True:
                generate_signals()
                logging.info("Waiting for next signal generation cycle...")
                time.sleep(3600)  # Run every hour
        except KeyboardInterrupt:
            logging.info("Signal generation interrupted by user")
    else:
        generate_signals()
        logging.info("Signal generation complete")

if __name__ == "__main__":
    main()
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    logging.info(f"Created signal generation script: {script_path}")


def main():
    """Main entry point for the process manager."""
    parser = argparse.ArgumentParser(description='Process Manager for Crypto Trading Platform')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status', 'monitor'],
                      help='Action to perform: start, stop, restart, status, or monitor')
    parser.add_argument('--process', help='Specific process to act on (for restart only)')
    
    args = parser.parse_args()
    
    manager = ProcessManager()
    
    try:
        # Create the signal generation script if needed
        create_generate_signals_script()
        
        if args.action == 'start':
            manager.start_all_processes()
            manager.print_status()
        
        elif args.action == 'stop':
            manager.stop_all_processes()
            print("All processes stopped.")
        
        elif args.action == 'restart':
            if args.process:
                if args.process in MANAGED_PROCESSES:
                    manager.restart_process(args.process)
                    print(f"Process {args.process} restarted.")
                else:
                    print(f"Unknown process: {args.process}")
                    print(f"Available processes: {', '.join(MANAGED_PROCESSES.keys())}")
            else:
                manager.stop_all_processes()
                time.sleep(2)  # Give processes time to stop
                manager.start_all_processes()
                manager.print_status()
        
        elif args.action == 'status':
            manager.print_status()
        
        elif args.action == 'monitor':
            manager.run_monitor()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        manager.cleanup()


if __name__ == "__main__":
    main()