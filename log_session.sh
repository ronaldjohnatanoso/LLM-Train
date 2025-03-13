#!/bin/bash
# filepath: /home/client/thes/log_session.sh

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/session_${TIMESTAMP}.log"

mkdir -p logs
script -a "$LOG_FILE"

echo "Log saved to: $LOG_FILE"