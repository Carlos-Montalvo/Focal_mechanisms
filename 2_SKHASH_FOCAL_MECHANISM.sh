#!/bin/bash

# Bash script to compute focal mechanisms using SKHASH for RPNet polarity results
# Processes multiple days in parallel for efficiency

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Directory parameters
BASEDIR="/Volumes/GeoPhysics_49/users-data/montalca"
CATLOGDIR="$BASEDIR/CATALOGS"
RPNET_DIR="$CATLOGDIR/RPNET"
POLARITIES_DIR="$RPNET_DIR/POLARITIES"
PARALLEL_JOBS=5  # Number of days to process in parallel

# Log directory
LOG_DIR="/Volumes/GeoPhysics_49/users-data/montalca/PROGRAMS/PYTHON/OUT_LOGS"
mkdir -p "$LOG_DIR"

# =============================================================================
# COLORS FOR OUTPUT
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Function to display configuration
display_config() {
    echo "=================================================="
    echo -e "${PURPLE}SKHASH FOCAL MECHANISM CONFIGURATION${NC}"
    echo "=================================================="
    echo -e "Base directory: ${YELLOW}$BASEDIR${NC}"
    echo -e "RPNET dir: ${YELLOW}$RPNET_DIR${NC}"
    echo -e "Polarities dir: ${YELLOW}$POLARITIES_DIR${NC}"
    echo -e "Parallel jobs: ${YELLOW}$PARALLEL_JOBS${NC}"
    echo -e "Log directory: ${YELLOW}$LOG_DIR${NC}"
    echo "=================================================="
}

# =============================================================================
# CONFIGURATION DISPLAY
# =============================================================================

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}    SKHASH FOCAL MECHANISM - MULTI-DAY PROCESSING${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

display_config
echo ""

# =============================================================================
# VALIDATION
# =============================================================================

echo -e "${YELLOW}Validating configuration...${NC}"

# Check if POLARITIES directory exists
if [ ! -d "$POLARITIES_DIR" ]; then
    echo -e "${RED}Error: Polarities directory $POLARITIES_DIR does not exist!${NC}"
    echo "Please run 1_RPNET_POLARITY_PICKER.sh first"
    exit 1
fi

echo -e "${GREEN}✓ Polarities directory exists${NC}"

# Check if SKHASH is available
if ! command -v SKHASH &> /dev/null; then
    echo -e "${RED}Error: SKHASH command not found!${NC}"
    echo "Please ensure SKHASH is installed and available in PATH"
    exit 1
fi

echo -e "${GREEN}✓ SKHASH is available${NC}"
echo ""

# =============================================================================
# FIND DATE DIRECTORIES
# =============================================================================

echo -e "${YELLOW}Searching for processing results (YYYY_JDD directories)...${NC}"

# Find all YYYY_JDD directories in POLARITIES (using glob pattern for compatibility)
DATE_DIRS=$(find "$POLARITIES_DIR" -maxdepth 1 -type d -name "[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9]" | sort)

if [ -z "$DATE_DIRS" ]; then
    echo -e "${RED}Error: No YYYY_JDD directories found in $POLARITIES_DIR${NC}"
    echo "Please ensure RPNet processing completed successfully"
    exit 1
fi

# Count total directories
TOTAL_DIRS=$(echo "$DATE_DIRS" | wc -l | tr -d ' ')
echo -e "${GREEN}✓ Found $TOTAL_DIRS day(s) to process${NC}"
echo ""

# Statistics
SUCCESSFUL_DIRS=0
FAILED_DIRS=0
FAILED_DIRS_LIST=()

# Start timing
SCRIPT_START_TIME=$(date +%s)
echo "Processing started at: $(date)"
echo ""

# =============================================================================
# PROCESSING FUNCTION
# =============================================================================

# Function to run SKHASH for a single day
process_skhash_day() {
    local day_dir="$1"
    local date_id=$(basename "$day_dir")
    local DAY_START_TIME=$(date +%s)
    local hash2_dir="$day_dir/hash2"
    local control_file="$hash2_dir/control_file.txt"
    
    echo -e "${YELLOW}[PID $$] Processing $date_id${NC}"
    
    # Check if hash2 directory exists
    if [ ! -d "$hash2_dir" ]; then
        echo -e "${RED}✗ [PID $$] $date_id: hash2 directory not found at $hash2_dir${NC}"
        echo "FAILED:$date_id" >> /tmp/skhash_results_$$.tmp
        return 1
    fi
    
    # Check if control_file exists
    if [ ! -f "$control_file" ]; then
        echo -e "${RED}✗ [PID $$] $date_id: control_file.txt not found at $control_file${NC}"
        echo "FAILED:$date_id" >> /tmp/skhash_results_$$.tmp
        return 1
    fi
    
    # Run SKHASH
    SKHASH "$control_file" 2>&1 | tee "$LOG_DIR/skhash_$date_id.log"
    
    local exit_code=$?
    local DAY_END_TIME=$(date +%s)
    local DAY_DURATION=$((DAY_END_TIME - DAY_START_TIME))
    local DAY_MINUTES=$((DAY_DURATION / 60))
    local DAY_SECONDS=$((DAY_DURATION % 60))
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ [PID $$] $date_id completed successfully in ${DAY_MINUTES}m ${DAY_SECONDS}s${NC}"
        echo "SUCCESS:$date_id:$DAY_DURATION" >> /tmp/skhash_results_$$.tmp
    else
        echo -e "${RED}✗ [PID $$] $date_id failed after ${DAY_MINUTES}m ${DAY_SECONDS}s (exit code: $exit_code)${NC}"
        echo "FAILED:$date_id:$DAY_DURATION" >> /tmp/skhash_results_$$.tmp
    fi
}

# Export function and variables for parallel execution
export -f process_skhash_day
export POLARITIES_DIR LOG_DIR RED GREEN YELLOW NC

# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

echo "Processing $TOTAL_DIRS day(s) in batches of $PARALLEL_JOBS..."
echo ""

# Clean up any previous temp files
rm -f /tmp/skhash_results_*.tmp

# Convert date directories to array for batch processing
mapfile -t DATE_ARRAY <<< "$DATE_DIRS"

# Process directories in parallel batches
batch_count=0
for ((i=0; i<${#DATE_ARRAY[@]}; i+=PARALLEL_JOBS)); do
    batch_count=$((batch_count + 1))
    batch_end=$((i + PARALLEL_JOBS - 1))
    if [ $batch_end -ge ${#DATE_ARRAY[@]} ]; then
        batch_end=$((${#DATE_ARRAY[@]} - 1))
    fi
    
    # Get date range for this batch
    first_date=$(basename "${DATE_ARRAY[$i]}")
    last_date=$(basename "${DATE_ARRAY[$batch_end]}")
    
    echo -e "${CYAN}=== Batch $batch_count: $first_date to $last_date ===${NC}"
    
    # Start parallel processes for this batch
    for ((j=i; j<=batch_end && j<${#DATE_ARRAY[@]}; j++)); do
        process_skhash_day "${DATE_ARRAY[$j]}" &
    done
    
    # Wait for all processes in this batch to complete
    wait
    
    echo -e "${CYAN}=== Batch $batch_count completed ===${NC}"
    echo ""
    
    # Small delay between batches to avoid overwhelming the system
    sleep 2
done

# =============================================================================
# RESULTS COLLECTION
# =============================================================================

# Calculate total processing time
SCRIPT_END_TIME=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

# Collect results from temporary files
SUCCESSFUL_DIRS=0
FAILED_DIRS=0
FAILED_DIRS_LIST=()
TOTAL_PROCESSING_TIME=0

for result_file in /tmp/skhash_results_*.tmp; do
    if [ -f "$result_file" ]; then
        while IFS=':' read -r status date_id duration; do
            if [ "$status" = "SUCCESS" ]; then
                ((SUCCESSFUL_DIRS++))
                [ -n "$duration" ] && TOTAL_PROCESSING_TIME=$((TOTAL_PROCESSING_TIME + duration))
            elif [ "$status" = "FAILED" ]; then
                ((FAILED_DIRS++))
                FAILED_DIRS_LIST+=("$date_id")
                [ -n "$duration" ] && TOTAL_PROCESSING_TIME=$((TOTAL_PROCESSING_TIME + duration))
            fi
        done < "$result_file"
    fi
done

# Clean up temporary files
rm -f /tmp/skhash_results_*.tmp

# =============================================================================
# FINAL SUMMARY
# =============================================================================

echo "=================================================="
echo -e "${PURPLE}FOCAL MECHANISM COMPUTATION COMPLETE${NC}"
echo "=================================================="
echo "Processing finished at: $(date)"
echo "Total wall-clock time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Total days processed: $TOTAL_DIRS"
echo -e "Successful days: ${GREEN}$SUCCESSFUL_DIRS${NC}"
echo -e "Failed days: ${RED}$FAILED_DIRS${NC}"
echo "SKHASH outputs saved to: $POLARITIES_DIR/*/hash2/"
echo "Log files saved to: $LOG_DIR/skhash_*.log"

if [ $FAILED_DIRS -gt 0 ]; then
    echo -e "${RED}Failed days:${NC}"
    for failed_date in "${FAILED_DIRS_LIST[@]}"; do
        echo -e "  ${RED}✗ $failed_date${NC}"
    done
    echo ""
fi

# Calculate averages if we have successful days
if [ $SUCCESSFUL_DIRS -gt 0 ]; then
    AVG_TIME_PER_DAY=$((TOTAL_PROCESSING_TIME / SUCCESSFUL_DIRS))
    AVG_MINUTES=$((AVG_TIME_PER_DAY / 60))
    AVG_SECONDS=$((AVG_TIME_PER_DAY % 60))
    echo "Average processing time per successful day: ${AVG_MINUTES}m ${AVG_SECONDS}s"
    
    # Calculate speedup achieved by parallel processing
    ESTIMATED_SEQUENTIAL_TIME=$((AVG_TIME_PER_DAY * SUCCESSFUL_DIRS))
    EST_SEQ_HOURS=$((ESTIMATED_SEQUENTIAL_TIME / 3600))
    EST_SEQ_MINUTES=$(((ESTIMATED_SEQUENTIAL_TIME % 3600) / 60))
    SPEEDUP=$(echo "scale=1; $ESTIMATED_SEQUENTIAL_TIME / $TOTAL_DURATION" | bc -l 2>/dev/null || echo "N/A")
    
    echo "Estimated sequential time would have been: ${EST_SEQ_HOURS}h ${EST_SEQ_MINUTES}m"
    if [ "$SPEEDUP" != "N/A" ]; then
        echo "Speedup achieved: ${SPEEDUP}x"
    fi
fi

echo "=================================================="

# Exit with appropriate code
if [ $FAILED_DIRS -gt 0 ]; then
    exit 1
else
    exit 0
fi
