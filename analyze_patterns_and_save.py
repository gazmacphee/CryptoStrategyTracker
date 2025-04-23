#!/usr/bin/env python3
"""
Script to analyze patterns across markets and save high-confidence recommendations.
This script can be run manually or scheduled to run hourly/daily.
"""

import os
import logging
import argparse
import subprocess
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pattern_analysis.log')
    ]
)
logger = logging.getLogger()

def run_pattern_analysis(output_file=None):
    """
    Run pattern analysis across all markets and optionally save results to a file.
    
    Args:
        output_file: Optional path to save results JSON (if None, just log results)
        
    Returns:
        Dictionary with analysis results or None if analysis failed
    """
    try:
        # Build the command
        cmd = [
            "python", "run_ml_fixed.py",
            "--analyze",
            "--verbose"
        ]
        
        # Run the command
        logger.info("Running pattern analysis across all markets")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log output
        for line in proc.stdout.splitlines():
            logger.info(f"  {line}")
            
        if proc.returncode != 0:
            logger.error("Pattern analysis failed")
            for line in proc.stderr.splitlines():
                logger.error(f"  {line}")
            return None
            
        # Try to extract results from stdout
        results = {}
        patterns_found = False
        pattern_count = 0
        
        for line in proc.stdout.splitlines():
            # Look for the line indicating patterns were found
            if "Found" in line and "patterns" in line:
                try:
                    # Extract pattern count with simple string parsing
                    parts = line.split("Found ")[1].split(" patterns")[0]
                    pattern_count = int(parts.strip())
                    patterns_found = True
                    results["pattern_count"] = pattern_count
                except:
                    pass
        
        # If we found patterns, provide more details
        if patterns_found:
            logger.info(f"Pattern analysis completed successfully with {pattern_count} patterns found")
            
            # Add timestamp
            results["timestamp"] = datetime.now().isoformat()
            
            # Save results to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Pattern analysis results saved to {output_file}")
                
            return results
        else:
            logger.warning("Pattern analysis completed but no patterns were found")
            return {"pattern_count": 0, "timestamp": datetime.now().isoformat()}
            
    except Exception as e:
        logger.error(f"Error running pattern analysis: {e}")
        return None

def save_trading_recommendations(min_strength=0.75):
    """
    Save high-confidence trading recommendations as signals.
    
    Args:
        min_strength: Minimum pattern strength to save (0.0-1.0)
        
    Returns:
        Number of recommendations saved or None if operation failed
    """
    try:
        # Build the command
        cmd = [
            "python", "run_ml_fixed.py",
            "--save",
            "--verbose"
        ]
        
        # Run the command
        logger.info(f"Saving trading recommendations with minimum strength {min_strength}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log output
        for line in proc.stdout.splitlines():
            logger.info(f"  {line}")
            
        if proc.returncode != 0:
            logger.error("Saving recommendations failed")
            for line in proc.stderr.splitlines():
                logger.error(f"  {line}")
            return None
            
        # Try to extract the number of saved signals
        saved_count = 0
        for line in proc.stdout.splitlines():
            if "Saved" in line and "trading signals" in line:
                try:
                    saved_count = int(line.split("Saved ")[1].split(" trading")[0].strip())
                except:
                    pass
                    
        logger.info(f"Successfully saved {saved_count} trading recommendations")
        return saved_count
        
    except Exception as e:
        logger.error(f"Error saving trading recommendations: {e}")
        return None

def run_analysis_and_save(output_file=None, min_strength=0.75, save_signals=True):
    """
    Run complete analysis workflow: detect patterns and save recommendations.
    
    Args:
        output_file: Optional path to save results JSON
        min_strength: Minimum pattern strength for recommendations
        save_signals: Whether to save signals to database
        
    Returns:
        Dictionary with workflow results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "analysis_success": False,
        "patterns_found": 0,
        "signals_saved": 0
    }
    
    # Run pattern analysis
    analysis_results = run_pattern_analysis(output_file)
    if analysis_results:
        results["analysis_success"] = True
        results["patterns_found"] = analysis_results.get("pattern_count", 0)
        
        # Save recommendations if requested and patterns were found
        if save_signals and results["patterns_found"] > 0:
            saved_count = save_trading_recommendations(min_strength)
            if saved_count is not None:
                results["signals_saved"] = saved_count
    
    logger.info(f"Analysis workflow completed: {results['patterns_found']} patterns found, {results['signals_saved']} signals saved")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze patterns and save recommendations")
    parser.add_argument("--output", default=None, help="Output file for analysis results")
    parser.add_argument("--min-strength", type=float, default=0.75, help="Minimum pattern strength (0.0-1.0)")
    parser.add_argument("--no-save", action="store_true", help="Don't save signals to database")
    args = parser.parse_args()
    
    logger.info("Starting pattern analysis workflow")
    
    # Run the workflow
    results = run_analysis_and_save(
        output_file=args.output,
        min_strength=args.min_strength,
        save_signals=not args.no_save
    )
    
    # Output summary
    if results["analysis_success"]:
        logger.info("Pattern analysis workflow completed successfully")
    else:
        logger.error("Pattern analysis workflow failed")