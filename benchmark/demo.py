"""
Demo Script - Chạy benchmark đầy đủ
"""

import asyncio
import os
import sys

# Add benchmark directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import BenchmarkSuite

async def run_demo():
    """Demo chạy benchmark"""
    print("🚀 SpectroChain-Dental Benchmark Demo")
    print("=" * 50)
    
    # Tạo benchmark suite
    benchmark = BenchmarkSuite()
    
    try:
        print("📊 Bắt đầu Performance Testing...")
        performance_results = await benchmark.run_performance_only()
        print(f"✅ Performance: {performance_results.get('summary', {}).get('grade', 'N/A')}")
        
        print("\n🔒 Bắt đầu Security Analysis...")
        security_results = await benchmark.run_security_only()
        security_score = security_results.get('overall_security_score', 0)
        print(f"✅ Security Score: {security_score:.3f}")
        
        print("\n🎯 Bắt đầu Accuracy Evaluation...")
        accuracy_results = await benchmark.run_accuracy_only()
        accuracy_grade = accuracy_results.get('summary', {}).get('grade', 'N/A')
        print(f"✅ Accuracy: {accuracy_grade}")
        
        print("\n📋 TỔNG KẾT:")
        print("-" * 30)
        print(f"Performance Grade: {performance_results.get('summary', {}).get('grade', 'N/A')}")
        print(f"Security Score: {security_score:.3f}")
        print(f"Accuracy Grade: {accuracy_grade}")
        
        # Tạo báo cáo
        print("\n📄 Đang tạo báo cáo HTML...")
        full_results = {
            'performance': performance_results,
            'security': security_results,
            'accuracy': accuracy_results
        }
        
        from utils.report_generator import ReportGenerator
        report_gen = ReportGenerator()
        report_path = await report_gen.generate_html_report(full_results)
        print(f"📊 Báo cáo đã được tạo: {report_path}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    asyncio.run(run_demo())
