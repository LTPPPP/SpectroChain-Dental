#!/usr/bin/env python3
"""
🚀 SpectroChain-Dental Evaluation Runner
=========================================

Main script for running comprehensive performance evaluation of SpectroChain-Dental.
All data is calculated in real-time with no hardcoded values.

Author: SpectroChain-Dental Team
Version: 2.0 (Real-Time Edition)
License: MIT
"""

import sys
import os
import argparse
import time
from pathlib import Path

def setup_paths():
    """Setup Python paths for module imports"""
    current_dir = Path(__file__).parent.absolute()
    
    # Add directories to Python path
    paths_to_add = [
        current_dir,
        current_dir / "src",
        current_dir / "benchmarks" / "real_time",
        current_dir / "evaluation" / "metrics",
        current_dir / "results" / "charts"
    ]
    
    for path in paths_to_add:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

def run_full_evaluation():
    """Run comprehensive evaluation with all metrics"""
    print("🚀 Starting SpectroChain-Dental Full Evaluation...")
    print("=" * 60)
    
    try:
        # Import and run performance metrics
        from evaluation.metrics.performance_metrics import PerformanceEvaluator
        
        evaluator = PerformanceEvaluator()
        evaluator.run_comprehensive_evaluation()
        
        print("\n✅ Evaluation completed successfully!")
        print("📁 Results saved to:")
        print("   • results/data/evaluation_results.json")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return False
    
    return True

def run_visualization():
    """Generate visualization charts"""
    print("\n🎨 Creating visualization charts...")
    print("-" * 40)
    
    try:
        from results.charts.visualization_charts import VisualizationCharts
        
        visualizer = VisualizationCharts()
        visualizer.generate_all_charts()
        
        print("\n✅ Charts created successfully!")
        print("📁 Charts saved to:")
        print("   • results/charts/*.png")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        return False
    
    return True

def run_benchmark_only():
    """Run only real-time benchmark"""
    print("🏁 Running real-time benchmark...")
    print("-" * 40)
    
    try:
        from benchmarks.real_time.benchmark_systems import RealTimeBenchmark
        
        benchmark = RealTimeBenchmark()
        results = benchmark.run_comprehensive_comparison()
        
        print("\n✅ Benchmark completed!")
        print("📊 Results preview:")
        for system, data in results.items():
            print(f"   • {system}: {data.get('overall_score', 0)}/100")
        
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        return False
    
    return True

def show_results():
    """Display quick results summary"""
    print("📊 Quick Results Summary")
    print("-" * 40)
    
    try:
        import json
        results_file = Path("results/data/evaluation_results.json")
        
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Display summary
            summary = results.get('evaluation_summary', {})
            comparative = results.get('comparative_analysis', {})
            
            print(f"🏆 Overall System Score: {summary.get('overall_system_score', 'N/A')}/100")
            print(f"🔗 Blockchain Score: {summary.get('blockchain_score', 'N/A')}/100")
            print(f"🎯 Verification Score: {summary.get('verification_score', 'N/A'):.1f}/100")
            print(f"🛡️  Security Score: {summary.get('security_score', 'N/A')}/100")
            
            print("\n📈 System Comparison:")
            for system, data in comparative.items():
                score = data.get('overall_score', 0)
                throughput = data.get('transaction_throughput_tps', 0)
                print(f"   • {system.replace('_', ' ').title()}: {score}/100 ({throughput:.1f} TPS)")
                
        else:
            print("❌ No results found. Run evaluation first!")
            
    except Exception as e:
        print(f"❌ Error reading results: {e}")

def print_banner():
    """Print application banner"""
    print("🏗️  SpectroChain-Dental Real-Time Evaluation System")
    print("=" * 60)
    print("📋 Features:")
    print("   ✅ 100% Real-time calculation (no hardcoded values)")
    print("   ✅ Multi-system benchmark (3 systems)")
    print("   ✅ Security penetration testing (STRIDE)")
    print("   ✅ Physical verification (spectral analysis)")
    print("   ✅ Professional visualization (5 charts)")
    print("=" * 60)

def print_completion_info(duration):
    """Print completion information and next steps"""
    print(f"\n⏱️  Total execution time: {duration:.2f} seconds")
    print("\n🎯 Next steps:")
    print("   📊 View charts: results/charts/*.png")
    print("   📄 Read data: results/data/evaluation_results.json")
    print("   📝 Check report: results/reports/REAL_TIME_BENCHMARK_SUMMARY.md")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="SpectroChain-Dental Real-Time Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py                    # Full evaluation + charts
  python run_evaluation.py --benchmark-only   # Only benchmark
  python run_evaluation.py --charts-only      # Only charts
  python run_evaluation.py --show-results     # Show latest results
        """
    )
    
    parser.add_argument('--benchmark-only', action='store_true',
                       help='Run only real-time benchmark')
    parser.add_argument('--charts-only', action='store_true',
                       help='Generate only visualization charts')
    parser.add_argument('--show-results', action='store_true',
                       help='Show latest evaluation results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup paths
    setup_paths()
    
    # Print banner
    print_banner()
    
    start_time = time.time()
    success = True
    
    try:
        if args.show_results:
            show_results()
        elif args.benchmark_only:
            success = run_benchmark_only()
        elif args.charts_only:
            success = run_visualization()
        else:
            # Full evaluation
            success = run_full_evaluation()
            if success:
                run_visualization()
    
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            print_completion_info(duration)
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 