"""
Script to check and clear the backfill lock file
"""

import os
from datetime import datetime

LOCK_FILE = ".backfill_lock"

def check_lock_status():
    """Check if backfill lock exists and when it was created"""
    if os.path.exists(LOCK_FILE):
        created_time = datetime.fromtimestamp(os.path.getctime(LOCK_FILE))
        print(f"Backfill lock exists - created at {created_time}")
        time_diff = datetime.now() - created_time
        print(f"Lock has been active for {time_diff}")
        return True
    else:
        print("No backfill lock found - no backfill is currently running")
        return False

def clear_lock():
    """Remove the backfill lock file"""
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
        print("Lock file removed successfully")
    else:
        print("No lock file to remove")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage backfill lock')
    parser.add_argument('--check', action='store_true', help='Check lock status')
    parser.add_argument('--clear', action='store_true', help='Clear lock file')
    
    args = parser.parse_args()
    
    if args.check:
        check_lock_status()
    elif args.clear:
        clear_lock()
    else:
        # If no arguments provided, do both
        print("Checking lock status:")
        is_locked = check_lock_status()
        
        if is_locked:
            print("\nDo you want to clear the lock? (y/n)")
            response = input().strip().lower()
            if response == 'y':
                clear_lock()