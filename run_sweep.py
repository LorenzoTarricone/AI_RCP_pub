#!/usr/bin/env python
"""
Hyperparameter sweep runner for the baseline VAE model using Weights & Biases.

Usage:
    python run_sweep.py --sweep_config configs/sweep_config.yaml --project baseline --count 10

This script will:
1. Initialize a W&B sweep with the provided configuration
2. Run sweep agents to execute hyperparameter optimization
3. Log results and find the best performing hyperparameters
"""

import argparse
import yaml
import wandb
import os
import subprocess
import sys
from pathlib import Path

def load_sweep_config(config_path):
    """Load the sweep configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            sweep_config = yaml.safe_load(f)
        return sweep_config
    except FileNotFoundError:
        print(f"Error: Sweep configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing sweep configuration YAML: {e}")
        sys.exit(1)

def initialize_sweep(sweep_config, project_name):
    """Initialize the W&B sweep and return sweep ID."""
    try:
        # Extract entity from sweep config if available
        entity = sweep_config.get('entity', None)
        if entity:
            sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
            print(f"✅ Sweep initialized successfully!")
            print(f"📊 Sweep ID: {sweep_id}")
            print(f"🔗 View sweep at: https://wandb.ai/{entity}/{project_name}/sweeps/{sweep_id}")
        else:
            sweep_id = wandb.sweep(sweep_config, project=project_name)
            print(f"✅ Sweep initialized successfully!")
            print(f"📊 Sweep ID: {sweep_id}")
            print(f"🔗 View sweep at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
        return sweep_id
    except Exception as e:
        print(f"❌ Error initializing sweep: {e}")
        sys.exit(1)

def run_sweep_agent(sweep_id, project_name, entity=None, count=None):
    """Run the sweep agent."""
    try:
        print(f"\n🚀 Starting sweep agent...")
        print(f"📈 Project: {project_name}")
        if entity:
            print(f"🏢 Entity: {entity}")
        print(f"🎯 Sweep ID: {sweep_id}")
        
        if count:
            print(f"🔢 Max runs: {count}")
            if entity:
                wandb.agent(sweep_id, project=project_name, entity=entity, count=count)
            else:
                wandb.agent(sweep_id, project=project_name, count=count)
        else:
            print("♾️  Running indefinitely (Ctrl+C to stop)")
            if entity:
                wandb.agent(sweep_id, project=project_name, entity=entity)
            else:
                wandb.agent(sweep_id, project=project_name)
            
    except KeyboardInterrupt:
        print("\n⏹️  Sweep interrupted by user")
    except Exception as e:
        print(f"❌ Error running sweep agent: {e}")

def print_best_run_info(project_name, sweep_id, entity=None):
    """Print information about the best run from the sweep."""
    try:
        api = wandb.Api()
        
        # Use provided entity or fall back to default
        entity_name = entity or api.default_entity
        sweep = api.sweep(f"{entity_name}/{project_name}/{sweep_id}")
        
        # Get the best run
        best_run = sweep.best_run()
        
        if best_run:
            print(f"\n🏆 Best Run Information:")
            print(f"   Run ID: {best_run.id}")
            print(f"   Run Name: {best_run.name}")
            print(f"   Best Score: {best_run.summary.get('mean_accuracy_pos', 'N/A')}")
            print(f"   Config: {best_run.config}")
            print(f"   URL: {best_run.url}")
        else:
            print("ℹ️  No completed runs found yet")
            
    except Exception as e:
        print(f"⚠️  Could not retrieve best run info: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run W&B hyperparameter sweep for baseline VAE model")
    parser.add_argument("--sweep_config", type=str, default="configs/sweep_config.yaml",
                       help="Path to sweep configuration YAML file")
    parser.add_argument("--project", type=str, default="baseline",
                       help="W&B project name for the sweep")
    parser.add_argument("--count", type=int, default=None,
                       help="Number of runs to execute (default: run indefinitely)")
    parser.add_argument("--init_only", action="store_true",
                       help="Only initialize the sweep, don't run agent")
    parser.add_argument("--agent_only", type=str, default=None,
                       help="Only run agent for existing sweep (provide sweep_id)")
    parser.add_argument("--entity", type=str, default=None,
                       help="W&B entity name (if not provided, will try to load from sweep config)")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    if not os.path.exists("train_gen_model.py"):
        print("❌ Error: train_gen_model.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if base config exists
    if not os.path.exists("configs/gen_config.yaml"):
        print("❌ Error: configs/gen_config.yaml not found.")
        sys.exit(1)
    
    print("🔬 W&B Hyperparameter Sweep Runner")
    print("=" * 50)
    
    if args.agent_only:
        # Run agent for existing sweep
        print(f"🔄 Running agent for existing sweep: {args.agent_only}")
        
        # Try to get entity from command line or sweep config
        entity = args.entity
        if not entity:
            # Try to load entity from sweep config
            try:
                print(f"📋 Loading entity from sweep configuration: {args.sweep_config}")
                sweep_config = load_sweep_config(args.sweep_config)
                entity = sweep_config.get('entity', None)
                if entity:
                    print(f"✅ Found entity in config: {entity}")
            except Exception as e:
                print(f"⚠️  Could not load sweep config for entity: {e}")
                print("🔄 Proceeding without explicit entity (will use default)")
        
        run_sweep_agent(args.agent_only, args.project, entity, args.count)
        print_best_run_info(args.project, args.agent_only, entity)
        
    else:
        # Load sweep configuration
        print(f"📋 Loading sweep configuration from: {args.sweep_config}")
        sweep_config = load_sweep_config(args.sweep_config)
        
        # Display sweep info
        print(f"🎯 Optimization method: {sweep_config['method']}")
        print(f"📊 Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
        print(f"🎛️  Parameters to optimize: {len(sweep_config['parameters'])}")
        
        # Initialize sweep
        sweep_id = initialize_sweep(sweep_config, args.project)
        
        if not args.init_only:
            # Run sweep agent
            entity = args.entity or sweep_config.get('entity', None)
            run_sweep_agent(sweep_id, args.project, entity, args.count)
            print_best_run_info(args.project, sweep_id, entity)
        else:
            print(f"\n✅ Sweep initialized. Run agents with:")
            entity_param = f" --entity {sweep_config.get('entity')}" if sweep_config.get('entity') else ""
            print(f"   python run_sweep.py --agent_only {sweep_id} --project {args.project}{entity_param}")

if __name__ == "__main__":
    main() 