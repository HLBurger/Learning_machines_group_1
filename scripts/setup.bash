#!/usr/bin/env bash
# Detect mode from arguments
# MODE="$1"

export ROS_MASTER_URI="http://172.20.10.13:11311"

# if [[ "$MODE" == "--hardware" ]]; then
#     # Hardware mode: connect to actual Robobo robot
#     export ROS_MASTER_URI="http://192.168.86.203:11311"
#     echo "Hardware mode: Connecting to Robobo at 10.122.37.237"
# else
#     # Simulation mode: use localhost
#     export ROS_MASTER_URI="http://localhost:11311"
#     echo "Simulation mode: Using localhost"
# fi

# You want your local IP, usually starting with 192.168, following RFC1918
# Windows powershell:
#    (Get-NetIPAddress | Where-Object { $_.AddressState -eq "Preferred" -and $_.ValidLifetime -lt "24:00:00" }).IPAddress
# linux:
#    hostname -I | awk '{print $1}'
# macOS:
#    ipconfig getifaddr en1
export COPPELIA_SIM_IP="192.168.86.75"
