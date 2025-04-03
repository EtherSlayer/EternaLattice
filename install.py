#!/usr/bin/env python3
"""
EternaLattice Installation Script

This script helps users set up and configure an EternaLattice node on their system.
It handles dependency installation, configuration, and initial setup.

Usage:
  python install.py [--datadir PATH] [--port PORT]
"""

import argparse
import os
import platform
import subprocess
import sys
import shutil
import sqlite3
from pathlib import Path

def check_python_version():
    """Check that Python version is at least 3.8."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"Error: EternaLattice requires Python {required_version[0]}.{required_version[1]} or higher")
        print(f"Current Python version is {current_version[0]}.{current_version[1]}")
        sys.exit(1)
    
    print(f"✓ Python version {'.'.join(map(str, current_version))} detected")

def install_dependencies():
    """Install required Python packages."""
    try:
        print("Installing dependencies...")
        requirements = [
            "flask",
            "flask-sqlalchemy",
            "matplotlib",
            "networkx",
            "numpy",
            "sqlalchemy"
        ]
        
        # Check if pip is available
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("Error: pip is not installed or not in PATH")
            sys.exit(1)
        
        # Install each package
        for package in requirements:
            print(f"  Installing {package}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", package],
                check=True
            )
        
        print("✓ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def setup_data_directory(data_dir):
    """Create and set up the data directory."""
    try:
        # Create main data directory
        os.makedirs(data_dir, exist_ok=True)
        print(f"✓ Data directory created at {data_dir}")
        
        # Create subdirectories
        subdirs = ["keys", "logs", "data"]
        for subdir in subdirs:
            path = os.path.join(data_dir, subdir)
            os.makedirs(path, exist_ok=True)
            print(f"  ✓ Created {subdir} directory")
        
        # Create empty SQLite database
        db_path = os.path.join(data_dir, "eternalattice.db")
        conn = sqlite3.connect(db_path)
        conn.close()
        print(f"✓ SQLite database initialized at {db_path}")
        
        return True
    except Exception as e:
        print(f"Error setting up data directory: {e}")
        return False

def create_config_file(data_dir, port):
    """Create a customized config file."""
    try:
        # Read the template config file
        config_path = os.path.join(os.getcwd(), "local_config.py")
        with open(config_path, "r") as f:
            config_content = f.read()
        
        # Customize content
        config_content = config_content.replace("DATA_DIR = os.path.join(HOME_DIR, \".eternalattice\")", 
                                              f"DATA_DIR = \"{data_dir}\"")
        config_content = config_content.replace("DEFAULT_PORT = 4444", 
                                              f"DEFAULT_PORT = {port}")
        
        # Write to user config file
        user_config_path = os.path.join(data_dir, "config.py")
        with open(user_config_path, "w") as f:
            f.write(config_content)
        
        print(f"✓ Configuration file created at {user_config_path}")
        return True
    except Exception as e:
        print(f"Error creating config file: {e}")
        return False

def create_launcher_script(data_dir):
    """Create a simple launcher script."""
    try:
        # Create launcher script using the standalone_main.py approach
        launcher_path = os.path.join(data_dir, "run_node.py")
        launcher_content = f'''#!/usr/bin/env python3
"""
EternaLattice Node Launcher

This script launches an EternaLattice node with your configuration.
"""

import os
import sys

# Add the EternaLattice directory to Python path
eternalattice_path = "{os.getcwd()}"
sys.path.insert(0, eternalattice_path)

# Run the standalone node
from standalone_main import main
if __name__ == "__main__":
    main()
'''
        
        with open(launcher_path, "w") as f:
            f.write(launcher_content)
        
        # Make executable on Unix-like systems
        if platform.system() != "Windows":
            os.chmod(launcher_path, 0o755)
        
        print(f"✓ Launcher script created at {launcher_path}")
        
        # Create batch/shell script for easier starting with improved parameters
        if platform.system() == "Windows":
            bat_path = os.path.join(data_dir, "run_node.bat")
            bat_content = f'''@echo off
echo Starting EternaLattice Node...
"{sys.executable}" "{launcher_path}" --data-dir="{data_dir}" %*
'''
            with open(bat_path, "w") as f:
                f.write(bat_content)
            print(f"✓ Windows batch file created at {bat_path}")
            
            # Create an additional script for running in standalone web mode
            web_bat_path = os.path.join(data_dir, "run_web_interface.bat")
            web_bat_content = f'''@echo off
echo Starting EternaLattice Web Interface...
"{sys.executable}" "{launcher_path}" --data-dir="{data_dir}" --port=5000 %*
'''
            with open(web_bat_path, "w") as f:
                f.write(web_bat_content)
            print(f"✓ Web interface batch file created at {web_bat_path}")
        else:
            sh_path = os.path.join(data_dir, "run_node.sh")
            sh_content = f'''#!/bin/bash
echo "Starting EternaLattice Node..."
"{sys.executable}" "{launcher_path}" --data-dir="{data_dir}" "$@"
'''
            with open(sh_path, "w") as f:
                f.write(sh_content)
            os.chmod(sh_path, 0o755)
            print(f"✓ Shell script created at {sh_path}")
            
            # Create an additional script for running in standalone web mode
            web_sh_path = os.path.join(data_dir, "run_web_interface.sh")
            web_sh_content = f'''#!/bin/bash
echo "Starting EternaLattice Web Interface..."
"{sys.executable}" "{launcher_path}" --data-dir="{data_dir}" --port=5000 "$@"
'''
            with open(web_sh_path, "w") as f:
                f.write(web_sh_content)
            os.chmod(web_sh_path, 0o755)
            print(f"✓ Web interface shell script created at {web_sh_path}")
        
        # Create CLI convenience scripts
        cli_commands = [
            ("add_shard", "Add a new memory shard to the network"),
            ("mine", "Mine a new block in the lattice"),
            ("status", "Display node status"),
            ("explore", "Explore the blockchain and shards"),
            ("search_shards", "Search for memory shards"),
            ("leaderboard", "Show the leaderboard of top contributors"),
            ("profile", "View or edit your user profile"),
            ("backup", "Backup or restore blockchain data"),
            ("diagnostics", "Run network diagnostics")
        ]
        
        # Create a tools directory
        tools_dir = os.path.join(data_dir, "tools")
        os.makedirs(tools_dir, exist_ok=True)
        
        for cmd, desc in cli_commands:
            if platform.system() == "Windows":
                tool_path = os.path.join(tools_dir, f"{cmd}.bat")
                tool_content = f'''@echo off
echo {desc}
"{sys.executable}" "{os.path.join(data_dir, 'run_node.py')}" {cmd} %*
'''
                with open(tool_path, "w") as f:
                    f.write(tool_content)
            else:
                tool_path = os.path.join(tools_dir, cmd)
                tool_content = f'''#!/bin/bash
echo "{desc}"
"{sys.executable}" "{os.path.join(data_dir, 'run_node.py')}" {cmd} "$@"
'''
                with open(tool_path, "w") as f:
                    f.write(tool_content)
                os.chmod(tool_path, 0o755)
        
        print(f"✓ CLI tool scripts created in {tools_dir}")
        
        return True
    except Exception as e:
        print(f"Error creating launcher script: {e}")
        return False

def create_readme(data_dir):
    """Create a README file with usage instructions."""
    try:
        readme_path = os.path.join(data_dir, "README.txt")
        readme_content = '''
ETERNALATTICE NODE
==================

Welcome to EternaLattice, a decentralized knowledge preservation network!

GETTING STARTED
--------------

To start your node:

On Windows:
  Double-click run_node.bat
  
On macOS/Linux:
  Open Terminal and run:
  ./run_node.sh
  
COMMAND LINE USAGE
-----------------

The node software supports several commands:

- Start your node:
  python run_node.py start [--web] [--port PORT] [--peers LIST]

- Add knowledge to the network:
  python run_node.py add_shard "Your knowledge here" [--category CATEGORY] [--region REGION]

- Mine a new block:
  python run_node.py mine

- View node status:
  python run_node.py status

- Explore the blockchain:
  python run_node.py explore [--block X,Y,Z] [--shard ID] [--verbose]

- List connected peers:
  python run_node.py peers

For more information, run:
  python run_node.py help

SUPPORT
-------

For help and support, please visit the EternaLattice community forum.

Thank you for contributing to the preservation of human knowledge!
'''
        
        with open(readme_path, "w") as f:
            f.write(readme_content)
        
        print(f"✓ README file created at {readme_path}")
        return True
    except Exception as e:
        print(f"Error creating README: {e}")
        return False

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="EternaLattice node installation script")
    parser.add_argument("--datadir", help="Data directory path (default: ~/.eternalattice)")
    parser.add_argument("--port", type=int, default=4444, help="Port for P2P network (default: 4444)")
    
    args = parser.parse_args()
    
    # Default data directory
    if args.datadir:
        data_dir = os.path.abspath(args.datadir)
    else:
        home_dir = os.path.expanduser("~")
        data_dir = os.path.join(home_dir, ".eternalattice")
    
    print("=" * 60)
    print("ETERNALATTICE NODE INSTALLATION")
    print("=" * 60)
    print(f"This script will set up an EternaLattice node on your system.")
    print(f"Data will be stored in: {data_dir}")
    print(f"P2P network port: {args.port}")
    print("=" * 60)
    
    # Confirm with user
    if input("Continue with installation? (y/n): ").lower() != 'y':
        print("Installation cancelled.")
        return
    
    print("\nChecking system requirements...")
    check_python_version()
    
    print("\nInstalling dependencies...")
    if not install_dependencies():
        print("Installation failed. Please fix the errors and try again.")
        return
    
    print("\nSetting up data directory...")
    if not setup_data_directory(data_dir):
        print("Installation failed. Please fix the errors and try again.")
        return
    
    print("\nCreating configuration...")
    if not create_config_file(data_dir, args.port):
        print("Installation failed. Please fix the errors and try again.")
        return
    
    print("\nCreating launcher scripts...")
    if not create_launcher_script(data_dir):
        print("Installation failed. Please fix the errors and try again.")
        return
    
    print("\nCreating documentation...")
    if not create_readme(data_dir):
        print("Installation failed. Please fix the errors and try again.")
        return
    
    print("\n" + "=" * 60)
    print("INSTALLATION COMPLETE!")
    print("=" * 60)
    print(f"EternaLattice has been installed to: {data_dir}")
    print("\nTo start your node, run:")
    
    if platform.system() == "Windows":
        print(f"  {os.path.join(data_dir, 'run_node.bat')}")
    else:
        print(f"  {os.path.join(data_dir, 'run_node.sh')}")
    
    print("\nThank you for contributing to the preservation of human knowledge!")
    print("=" * 60)

if __name__ == "__main__":
    main()