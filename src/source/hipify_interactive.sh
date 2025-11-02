#!/bin/bash

# Interactive Hipify Script for DualSPHysics
# This script processes CUDA files one by one, waiting for user input after each file
# Usage: 
#   ./hipify_interactive.sh           - Interactive mode (asks for each file)
#   ./hipify_interactive.sh --auto    - Auto mode (processes all files without prompts)
#   CUDA_PATH=/path/to/cuda ./hipify_interactive.sh  - Custom CUDA path

# Default to CUDA 12.3
# You can override with: CUDA_PATH=/usr/local/cuda-12.8 ./hipify_interactive.sh
CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-12.3}

# Clang resource directory (optional, for newer Clang versions)
# Example: CLANG_RESOURCE_DIR=/opt/llvm-20.1.8/lib/clang/20
CLANG_RESOURCE_DIR=${CLANG_RESOURCE_DIR:-}

AUTO_MODE=false

# Parse arguments
if [ "$1" == "--auto" ] || [ "$1" == "-a" ]; then
    AUTO_MODE=true
fi

echo "=============================================="
echo "Interactive DualSPHysics CUDA to HIP Converter"
echo "=============================================="
echo ""
if [ "$AUTO_MODE" = true ]; then
    echo "MODE: AUTO (non-blocking, processes all files)"
else
    echo "MODE: INTERACTIVE (waits for confirmation)"
fi
echo "CUDA Path: $CUDA_PATH"
echo "Working Directory: $(pwd)"
echo ""
echo "NOTE: hipify-clang fully supports CUDA up to 12.3"
echo "      Using newer versions may produce warnings"
echo ""

# Check if hipify-clang is available
if ! command -v hipify-clang &> /dev/null; then
    echo "ERROR: hipify-clang not found. Please install it first."
    exit 1
fi

# Check if CUDA path exists
if [ ! -d "$CUDA_PATH" ]; then
    echo "WARNING: CUDA path $CUDA_PATH not found."
    echo "You may need to adjust CUDA_PATH in this script."
    read -p "Continue anyway? (y/n): " continue_choice
    if [ "$continue_choice" != "y" ]; then
        exit 1
    fi
fi

echo ""
echo "Scanning for CUDA files..."
echo ""

# Find all .cu and .cuh files
cu_files=($(find . -maxdepth 1 -name "*.cu" -type f | sort))
cuh_files=($(find . -maxdepth 1 -name "*.cuh" -type f | sort))

total_files=$((${#cu_files[@]} + ${#cuh_files[@]}))

if [ $total_files -eq 0 ]; then
    echo "No CUDA files (.cu or .cuh) found in current directory."
    exit 0
fi

echo "Found ${#cu_files[@]} .cu files and ${#cuh_files[@]} .cuh files"
echo "Total: $total_files files to process"
echo ""

current_file=0

# Function to process a single file
process_file() {
    local file=$1
    local file_num=$2
    local total=$3
    
    echo "=============================================="
    echo "File $file_num of $total: $file"
    echo "=============================================="
    
    # Show file info
    echo ""
    echo "File size: $(wc -c < "$file") bytes"
    echo "Lines: $(wc -l < "$file") lines"
    echo ""
    
    local choice="y"
    
    if [ "$AUTO_MODE" = false ]; then
        read -p "Process this file? (y/n/s/q): " choice
        echo ""
    else
        echo "AUTO MODE: Processing automatically..."
        echo ""
    fi
    
    case $choice in
        y|Y)
            # Determine output filename
            local output_file
            if [[ "$file" == *.cu ]]; then
                output_file="${file%.cu}.hip.cpp"
            elif [[ "$file" == *.cuh ]]; then
                output_file="${file%.cuh}.hip.h"
            else
                output_file="${file}.hip"
            fi
            
            echo "Running hipify-clang on $file..."
            echo "Output: $output_file"
            echo ""
            
            # Build hipify command
            local hipify_cmd="hipify-clang \"$file\" \
                --cuda-path=$CUDA_PATH \
                -I. \
                -I$CUDA_PATH/include \
                --print-stats \
                -o \"$output_file\""
            
            # Add Clang resource directory if specified
            if [ -n "$CLANG_RESOURCE_DIR" ]; then
                hipify_cmd="$hipify_cmd --clang-resource-directory=$CLANG_RESOURCE_DIR"
            fi
            
            # Execute hipify
            eval $hipify_cmd
            
            exit_code=$?
            echo ""
            
            if [ $exit_code -eq 0 ]; then
                echo "✓ Successfully processed $file → $output_file"
            else
                echo "✗ Error processing $file (exit code: $exit_code)"
                if [ "$AUTO_MODE" = false ]; then
                    read -p "Continue with next file? (y/n): " continue_after_error
                    if [ "$continue_after_error" != "y" ]; then
                        echo "Stopping."
                        exit 1
                    fi
                else
                    echo "AUTO MODE: Continuing despite error..."
                fi
            fi
            ;;
        s|S)
            echo "⊘ Skipped $file"
            ;;
        q|Q)
            echo "Quitting."
            exit 0
            ;;
        *)
            echo "⊘ Skipped $file (invalid choice)"
            ;;
    esac
    
    echo ""
}

# Process .cu files
echo "Processing .cu files..."
echo ""
for file in "${cu_files[@]}"; do
    ((current_file++))
    process_file "$file" $current_file $total_files
done

# Process .cuh files
echo "Processing .cuh header files..."
echo ""
for file in "${cuh_files[@]}"; do
    ((current_file++))
    process_file "$file" $current_file $total_files
done

echo "=============================================="
echo "Hipify process complete!"
echo "Processed $current_file files"
echo "=============================================="
