#!/bin/bash
#SBATCH --job-name=interactive_session
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu-best
#SBATCH --nodelist=margpu005
#SBATCH --gres=gpu
#SBATCH --output=interactive_%j.out
#SBATCH --error=interactive_%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=nacim.belkhir@inria.fr

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Working directory: $(pwd)"
echo ""

# Print SSH connection information
echo "=========================================="
echo "SSH CONNECTION INFORMATION"
echo "=========================================="
echo "To connect to this allocated node, use:"
echo "ssh $(whoami)@$(hostname)"
echo ""
echo "Or from the login node:"
echo "ssh $USER@$SLURM_JOB_NODELIST"
echo ""
echo "Job will run until: $(date -d \"+${SLURM_JOB_TIME_LIMIT} minutes\")"
echo "=========================================="
echo ""

# Keep the job alive and show resource usage periodically
echo "Job is running. You can now SSH to the allocated node."
echo "Press Ctrl+C or scancel $SLURM_JOB_ID to terminate."
echo ""

# Function to show resource usage
show_usage() {
    echo "=== Resource Usage at $(date) ==="
    echo "Memory usage:"
    free -h
    echo ""
    echo "CPU usage:"
    top -bn1 | grep "Cpu(s)" | head -1
    echo ""
    echo "Load average:"
    uptime
    echo "================================="
    echo ""
}

# Trap to handle job termination gracefully
cleanup() {
    echo ""
    echo "Job terminating at: $(date)"
    echo "Total runtime: $SECONDS seconds"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main loop - keep job alive and show periodic updates
counter=0
while true; do
    sleep 1200  # Sleep for 20 minutes
    counter=$((counter + 1))
    
    echo "=== Status Update #$counter at $(date) ==="
    echo "Job still running on node: $(hostname)"
    echo "Time remaining: approximately $((SLURM_JOB_TIME_LIMIT - SECONDS/60)) minutes"
    
    # Show resource usage every 30 minutes (6 iterations)
    if [ $((counter % 6)) -eq 0 ]; then
        show_usage
    fi
    
    echo ""
done
