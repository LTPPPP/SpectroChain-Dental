#!/bin/bash

# SpectroChain-Dental Benchmark Runner Script
# Chạy benchmark với các options khác nhau

echo "🔬 SpectroChain-Dental Benchmark Suite"
echo "======================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Function to run specific benchmark
run_benchmark() {
    local mode=$1
    local config=${2:-benchmark_config.json}
    
    echo "🚀 Running $mode benchmark..."
    python main.py --mode $mode --config $config
    
    if [ $? -eq 0 ]; then
        echo "✅ $mode benchmark completed successfully"
    else
        echo "❌ $mode benchmark failed"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-full}" in
    "performance")
        run_benchmark "performance"
        ;;
    "security")
        run_benchmark "security"
        ;;
    "accuracy")
        run_benchmark "accuracy"
        ;;
    "full")
        echo "🎯 Running full benchmark suite..."
        run_benchmark "full"
        ;;
    "demo")
        echo "🎭 Running demo..."
        python demo.py
        ;;
    "help")
        echo "Usage: $0 [performance|security|accuracy|full|demo|help]"
        echo ""
        echo "Options:"
        echo "  performance  - Run only performance tests"
        echo "  security     - Run only security analysis"
        echo "  accuracy     - Run only accuracy evaluation"
        echo "  full         - Run complete benchmark suite (default)"
        echo "  demo         - Run demonstration script"
        echo "  help         - Show this help message"
        ;;
    *)
        echo "❓ Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

echo ""
echo "📊 Benchmark completed! Check the results/ directory for reports."
echo "🌐 Open the HTML report in your browser for detailed analysis."
