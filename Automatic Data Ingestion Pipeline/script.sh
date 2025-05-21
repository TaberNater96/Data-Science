#!/bin/bash

# Define directories
DEV_DIR="/dev"
SRC_DIR="/src"

# Define files
PYTHON_SCRIPT="$DEV_DIR/data_pipeline.py"
CHANGELOG="$DEV_DIR/changelog.txt"
CSV_FILE="$DEV_DIR/cademy_stats.csv"
DB_FILE="$DEV_DIR/cademycode.db"

# Function to get the latest version from changelog, takes the last line from the grep search,
# splits the line by spaces and takes the second field, then remove any colons from the result
get_latest_version() {
    if [ -f "$CHANGELOG" ]; then
        VERSION=$(grep "Version" "$CHANGELOG" | tail -n 1 | cut -d' ' -f2 | tr -d ':')
        echo "$VERSION"
    else
        echo "0.0"
    fi
}

# Get the current version before running the script
CURRENT_VERSION=$(get_latest_version)

# Run the Python script
python "$PYTHON_SCRIPT"

# Checks the exit status of the last executed command, if it equals 0, then it was successful
if [ $? -eq 0 ]; then
    echo "Data pipeline script executed successfully."
    
    # Get the new version after running the script
    NEW_VERSION=$(get_latest_version)
    
    # Check if there's an update by comparing new version with current version
    if [ "$NEW_VERSION" != "$CURRENT_VERSION" ]; then
        echo "Update detected. Moving files to production..."
        
        # Move CSV file to production
        mv "$CSV_FILE" "$SRC_DIR/"
        
        # Copy DB file to production (using cp to keep a copy in dev)
        cp "$DB_FILE" "$SRC_DIR/"
        
        # Copy changelog to production
        cp "$CHANGELOG" "$SRC_DIR/"
        
        echo "Files moved to production successfully."
        echo "Update completed. New version: $NEW_VERSION"
    else
        echo "No update detected. Production files remain unchanged."
    fi
else
    echo "Error occurred while executing the data pipeline script. Check error logs for details."
fi