#!/bin/bash
# TAO Docker Compose Runner with Configurable Settings
# ====================================================

set -e

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  up              Start all services (default: TAO API only)"
    echo "  up-all          Start all services including SeaweedFS"
    echo "  down            Stop all services"
    echo "  restart         Restart all services"
    echo "  logs            Show logs for all services"
    echo "  status          Show status of all services"
    echo "  config          Show current configuration"
    echo "  cleanup-seaweed Delete all objects in Seaweed tao-storage bucket (files only; workspaces stay in MongoDB)"
    echo "  clear-mongo     Clear MongoDB (removes workspaces, experiments, jobs). Use with 'cleanup-seaweed' for full reset."
    echo "  help            Show this help message"
    echo ""
    echo "Options:"
    echo "  --airgapped     Enable airgapped mode"
    echo "  --no-airgapped  Disable airgapped mode"
    echo "  --ptm           Enable pretrained models pull"
    echo "  --no-ptm        Disable pretrained models pull"
    echo "  --python=VER    Set Python version (e.g., --python=3.11)"
    echo "  --config=FILE   Use custom config file (default: config.env)"
    echo "  --clear_mongo   Clear MongoDB databases when used with 'down' command"
    echo ""
    echo "Examples:"
    echo "  $0 up                    # Start TAO API services only"
    echo "  $0 up-all                # Start all services including SeaweedFS"
    echo "  $0 up --airgapped        # Start in airgapped mode"
    echo "  $0 up --ptm              # Start with pretrained models pull"
    echo "  $0 up --python=3.11      # Use Python 3.11"
    echo "  $0 down --clear_mongo    # Stop services and clear MongoDB databases"
    echo "  $0 config                # Show current configuration"
    echo "  $0 cleanup-seaweed      # Remove all files in Seaweed (tao-storage bucket)"
    echo "  $0 clear-mongo          # Clear workspaces/experiments/jobs in MongoDB (services must be up)"
    echo "  $0 cleanup-seaweed && $0 clear-mongo   # Full reset: Seaweed + MongoDB"
}

# Default values
COMMAND="up"
CONFIG_FILE="config.env"
AIRGAPPED_OVERRIDE=""
PTM_OVERRIDE=""
PYTHON_OVERRIDE=""
SEAWEEDFS_PROFILE=""
CLEAR_MONGO=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        up|down|restart|logs|status|config|cleanup-seaweed|clear-mongo|help)
            COMMAND="$1"
            shift
            ;;
        up-all)
            COMMAND="up"
            SEAWEEDFS_PROFILE="--profile seaweedfs"
            shift
            ;;
        --airgapped)
            AIRGAPPED_OVERRIDE="AIRGAPPED_MODE=true"
            shift
            ;;
        --no-airgapped)
            AIRGAPPED_OVERRIDE="AIRGAPPED_MODE=false"
            shift
            ;;
        --ptm)
            PTM_OVERRIDE="PTM_PULL=true"
            shift
            ;;
        --no-ptm)
            PTM_OVERRIDE="PTM_PULL=false"
            shift
            ;;
        --python=*)
            PYTHON_OVERRIDE="PYTHON_VERSION=${1#*=}"
            shift
            ;;
        --config=*)
            CONFIG_FILE="${1#*=}"
            shift
            ;;
        --clear_mongo)
            CLEAR_MONGO=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    echo "Please copy config.env.example to $CONFIG_FILE and adjust settings as needed."
    exit 1
fi

# Source the config file
echo "Loading configuration from $CONFIG_FILE..."
set -a  # automatically export all variables
source "$CONFIG_FILE"
set +a

# Apply command line overrides
if [[ -n "$AIRGAPPED_OVERRIDE" ]]; then
    export $AIRGAPPED_OVERRIDE
    echo "Override: $AIRGAPPED_OVERRIDE"
fi

if [[ -n "$PTM_OVERRIDE" ]]; then
    export $PTM_OVERRIDE
    echo "Override: $PTM_OVERRIDE"
fi

if [[ -n "$PYTHON_OVERRIDE" ]]; then
    export $PYTHON_OVERRIDE
    echo "Override: $PYTHON_OVERRIDE"
fi

# Enforce airgapped mode logic
if [[ "$AIRGAPPED_MODE" == "true" ]]; then
    # No internet access means no PTM pull
    if [[ "$PTM_PULL" == "true" ]]; then
        echo "Warning: PTM_PULL disabled because AIRGAPPED_MODE=true (no internet access)"
        export PTM_PULL=false
    fi

    # No internet access means local storage is required
    if [[ "$SEAWEEDFS_ENABLED" == "false" ]]; then
        echo "Warning: SEAWEEDFS_ENABLED forced to true because AIRGAPPED_MODE=true (local storage required)"
        export SEAWEEDFS_ENABLED=true
    fi
fi

# Validate GPU configuration
validate_gpu_config() {
    # Skip validation for config and help commands
    if [[ "$COMMAND" == "config" ]] || [[ "$COMMAND" == "help" ]]; then
        return 0
    fi

    # Only validate on 'up' command
    if [[ "$COMMAND" != "up" ]]; then
        return 0
    fi

    echo ""
    echo "Validating GPU configuration..."

    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found. Cannot validate GPU count."
        echo "  If you have GPUs, please ensure NVIDIA drivers are installed."
        return 0
    fi

    # Get actual GPU count from system (excluding display/management GPUs)
    # Filter out "Display" GPUs which are not for compute workloads
    ACTUAL_GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -iv "display" | wc -l)

    if [[ $? -ne 0 ]] || [[ -z "$ACTUAL_GPU_COUNT" ]]; then
        echo "Warning: Failed to detect GPU count. nvidia-smi command failed."
        echo "  Proceeding with configuration value: NUM_GPU_PER_NODE=$NUM_GPU_PER_NODE"
        return 0
    fi

    # Get configured GPU count
    CONFIGURED_GPU_COUNT="${NUM_GPU_PER_NODE:-0}"

    echo "  Detected GPUs: $ACTUAL_GPU_COUNT (via nvidia-smi)"
    echo "  Configured NUM_GPU_PER_NODE: $CONFIGURED_GPU_COUNT"

    # Validate configuration matches reality
    if [[ "$ACTUAL_GPU_COUNT" -ne "$CONFIGURED_GPU_COUNT" ]]; then
        echo ""
        echo "ERROR: GPU count mismatch!"
        echo "========================================"
        echo "  System has $ACTUAL_GPU_COUNT GPU(s) but NUM_GPU_PER_NODE is set to $CONFIGURED_GPU_COUNT"
        echo ""
        echo "This mismatch will cause job allocation errors:"
        echo "  - If NUM_GPU_PER_NODE > actual GPUs: Jobs will pass validation but fail allocation"
        echo "  - If NUM_GPU_PER_NODE < actual GPUs: Some GPUs will be unused"
        echo ""
        echo "To fix this issue:"
        echo "  1. Edit $CONFIG_FILE"
        echo "  2. Set NUM_GPU_PER_NODE=$ACTUAL_GPU_COUNT"
        echo "  3. Run this script again"
        echo ""
        echo "Detected GPUs:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null
        echo ""
        return 1
    fi

    echo "  ✓ GPU configuration is correct"
    echo ""
    return 0
}

# Run GPU validation (will exit if validation fails)
if ! validate_gpu_config; then
    exit 1
fi

# Show current configuration if requested
if [[ "$COMMAND" == "config" ]]; then
    echo ""
    echo "Current Configuration:"
    echo "====================="
    echo "AIRGAPPED_MODE: $AIRGAPPED_MODE (controls internet access features)"
    echo "PTM_PULL: $PTM_PULL"
    echo "SEAWEEDFS_ENABLED: $SEAWEEDFS_ENABLED (controls local storage)"
    echo "PYTHON_VERSION: $PYTHON_VERSION"
    echo "DEPLOYMENT_MODE: $DEPLOYMENT_MODE"
    echo "IMAGE_TAO_PYTORCH: $IMAGE_TAO_PYTORCH"
    echo "DOCKER_NETWORK: $DOCKER_NETWORK"
    echo ""
    echo "To modify these settings:"
    echo "1. Edit $CONFIG_FILE"
    echo "2. Use command line overrides (--airgapped, --ptm, --python=X.Y)"
    echo "3. Set environment variables directly"
    exit 0
fi

# Show help if requested
if [[ "$COMMAND" == "help" ]]; then
    show_usage
    exit 0
fi

# Configure SeaweedFS environment variable for nginx
configure_seaweedfs_config() {
    if [[ "$SEAWEEDFS_ENABLED" == "true" ]]; then
        echo "Configuring nginx with SeaweedFS support..."
        export SEAWEEDFS_CONFIG='
    # SeaweedFS routes
    location /seaweedfs/master/ {
        set $seaweedfs_master http://seaweedfs-master:9333;
        proxy_pass $seaweedfs_master;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /seaweedfs/filer/ {
        set $seaweedfs_filer http://seaweedfs-filer:8888;
        proxy_pass $seaweedfs_filer;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /seaweedfs/s3/ {
        set $seaweedfs_s3 http://seaweedfs-s3:8333;
        proxy_pass $seaweedfs_s3;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }'
    else
        echo "Configuring nginx without SeaweedFS (SeaweedFS disabled)..."
        export SEAWEEDFS_CONFIG=""
    fi
}

# Build profile arguments
PROFILE_ARGS=""
if [[ -n "$SEAWEEDFS_PROFILE" ]] || [[ "$SEAWEEDFS_ENABLED" == "true" ]]; then
    PROFILE_ARGS="$PROFILE_ARGS --profile seaweedfs"
fi
if [[ "$PTM_PULL" == "true" ]]; then
    PROFILE_ARGS="$PROFILE_ARGS --profile ptm"
fi

# Execute docker compose command
echo ""
echo "Executing: docker compose $PROFILE_ARGS $COMMAND"
echo "Configuration loaded from: $CONFIG_FILE"
echo "AIRGAPPED_MODE: $AIRGAPPED_MODE"
echo "PTM_PULL: $PTM_PULL"
echo "PYTHON_VERSION: $PYTHON_VERSION"
echo ""

case $COMMAND in
    up)
        configure_seaweedfs_config
        docker compose $PROFILE_ARGS up -d
        echo ""
        echo "Services started successfully!"
        echo "TAO API: http://localhost"
        if [[ -n "$SEAWEEDFS_PROFILE" ]]; then
            echo "SeaweedFS Master: http://localhost:9333"
            echo "SeaweedFS Filer: http://localhost:8888"
            echo "SeaweedFS S3: http://localhost:8333"
        fi
        if [[ "$PTM_PULL" == "true" ]]; then
            echo "Pretrained models will be pulled on startup"
        fi
        ;;
    down)
        echo "Stopping job containers on tao_default network..."
        # Get all containers on the tao_default network
        NETWORK_CONTAINERS=$(docker ps -aq --filter "network=tao_default" 2>/dev/null)
        if [[ -n "$NETWORK_CONTAINERS" ]]; then
            # Filter out compose-managed containers
            for cid in $NETWORK_CONTAINERS; do
                # Check if container has compose project label (compose-managed)
                COMPOSE_PROJECT=$(docker inspect "$cid" --format='{{index .Config.Labels "com.docker.compose.project"}}' 2>/dev/null || echo "")
                if [[ -z "$COMPOSE_PROJECT" ]]; then
                    # Not compose-managed, so it's a job container
                    CONTAINER_NAME=$(docker inspect "$cid" --format='{{.Name}}' 2>/dev/null | sed 's/\///')
                    CONTAINER_IMAGE=$(docker inspect "$cid" --format='{{.Config.Image}}' 2>/dev/null)
                    echo "  Removing job container: $CONTAINER_NAME ($CONTAINER_IMAGE)"
                    docker rm -f "$cid" 2>/dev/null
                fi
            done
        else
            echo "  No job containers found on tao_default network."
        fi
        if [[ "$CLEAR_MONGO" == "true" ]]; then
            echo "Clearing MongoDB databases..."
            if docker ps --format '{{.Names}}' | grep -q "^mongodb$"; then
                docker exec -it mongodb mongosh -u default-user -p mongosecret --authenticationDatabase admin --eval "db.adminCommand('listDatabases').databases.forEach(function(d) { if (d.name !== 'admin' && d.name !== 'local' && d.name !== 'config') { db.getSiblingDB(d.name).dropDatabase(); } });" 2>/dev/null
                if [[ $? -eq 0 ]]; then
                    echo "  MongoDB databases cleared successfully."
                else
                    echo "  Warning: Failed to clear MongoDB databases. Container may not be running."
                fi
            else
                echo "  Warning: MongoDB container not running. Skipping database cleanup."
            fi
        fi
        echo "Stopping all docker-compose services (including SeaweedFS and PTM profiles)..."
        docker compose --profile seaweedfs --profile ptm down
        echo "All services stopped successfully."
        ;;
    restart)
        configure_seaweedfs_config
        docker compose $PROFILE_ARGS restart
        echo "Services restarted."
        ;;
    logs)
        docker compose $PROFILE_ARGS logs -f
        ;;
    status)
        docker compose $PROFILE_ARGS ps
        ;;
    cleanup-seaweed)
        SEAWEED_ENDPOINT="${SEAWEED_ENDPOINT:-http://localhost:8333}"
        export SEAWEED_ENDPOINT
        export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-seaweedfs}"
        export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-seaweedfs123}"
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        if [[ -x "$SCRIPT_DIR/cleanup-seaweed-storage.sh" ]]; then
            "$SCRIPT_DIR/cleanup-seaweed-storage.sh"
        else
            bash "$SCRIPT_DIR/cleanup-seaweed-storage.sh"
        fi
        ;;
    clear-mongo)
        echo "Clearing MongoDB (workspaces, experiments, jobs)..."
        if docker ps --format '{{.Names}}' | grep -q "^mongodb$"; then
            if docker exec mongodb mongosh -u default-user -p mongosecret --authenticationDatabase admin --eval "db.adminCommand('listDatabases').databases.forEach(function(d) { if (d.name !== 'admin' && d.name !== 'local' && d.name !== 'config') { db.getSiblingDB(d.name).dropDatabase(); } });" 2>/dev/null; then
                echo "  MongoDB cleared successfully. Workspaces and experiments are removed."
            else
                echo "  Warning: Failed to clear MongoDB."
                exit 1
            fi
        else
            echo "  Error: MongoDB container not running. Start services first: $0 up"
            exit 1
        fi
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac