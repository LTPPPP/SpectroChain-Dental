"""
SpectroChain-Dental Benchmark Suite
Hệ thống đánh giá toàn diện cho hiệu năng, bảo mật và độ chính xác
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from performance.performance_tester import PerformanceTester
    from security.security_analyzer import SecurityAnalyzer
    from accuracy.accuracy_evaluator import AccuracyEvaluator
    from utils.metrics_collector import MetricsCollector
    from utils.report_generator import ReportGenerator
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please install required packages: pip install -r requirements.txt")
    raise

class BenchmarkSuite:
    """Bộ công cụ benchmark tổng hợp"""
    
    def __init__(self, config_path: str = "benchmark_config.json"):
        self.config = self._load_config(config_path)
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Initialize components
        self.performance_tester = PerformanceTester(self.config.get('performance', {}))
        self.security_analyzer = SecurityAnalyzer(self.config.get('security', {}))
        self.accuracy_evaluator = AccuracyEvaluator(self.config.get('accuracy', {}))
        self.metrics_collector = MetricsCollector()
        self.report_generator = ReportGenerator()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load benchmark configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using default config")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default benchmark configuration"""
        return {
            "performance": {
                "blockchain_url": "http://127.0.0.1:7545",
                "backend_url": "http://127.0.0.1:8000",
                "test_duration": 300,  # 5 minutes
                "concurrent_users": [1, 5, 10, 20, 50],
                "test_scenarios": ["register", "verify", "mixed"]
            },
            "security": {
                "attack_types": ["selfish_mining", "double_spending", "eclipse"],
                "threshold_tests": [0.1, 0.25, 0.33, 0.5],
                "network_analysis": True
            },
            "accuracy": {
                "test_data_path": "../data/",
                "models": ["svm", "random_forest", "neural_network"],
                "cross_validation_folds": 5,
                "metrics": ["precision", "recall", "f1", "accuracy"]
            },
            "output": {
                "results_dir": "results",
                "generate_html_report": True,
                "generate_pdf_report": True,
                "save_raw_data": True
            }
        }
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Chạy tất cả các test benchmark"""
        logger.info("🚀 Bắt đầu benchmark toàn diện SpectroChain-Dental")
        self.start_time = time.time()
        
        try:
            # 1. Performance Testing
            logger.info("📊 Đang chạy Performance Testing...")
            performance_results = await self.performance_tester.run_all_tests()
            self.results['performance'] = performance_results
            
            # 2. Security Analysis
            logger.info("🔒 Đang chạy Security Analysis...")
            security_results = await self.security_analyzer.run_all_tests()
            self.results['security'] = security_results
            
            # 3. Accuracy Evaluation
            logger.info("🎯 Đang chạy Accuracy Evaluation...")
            accuracy_results = await self.accuracy_evaluator.run_all_tests()
            self.results['accuracy'] = accuracy_results
            
            # 4. Collect System Metrics
            logger.info("📈 Thu thập System Metrics...")
            system_metrics = self.metrics_collector.collect_system_metrics()
            self.results['system_metrics'] = system_metrics
            
            self.end_time = time.time()
            self.results['benchmark_info'] = {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(self.end_time).isoformat(),
                'duration_seconds': self.end_time - self.start_time,
                'config': self.config
            }
            
            # 5. Generate Reports
            logger.info("📄 Tạo báo cáo...")
            await self.generate_reports()
            
            logger.info("✅ Benchmark hoàn thành!")
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong quá trình benchmark: {e}")
            raise
    
    async def run_performance_only(self) -> Dict[str, Any]:
        """Chỉ chạy performance testing"""
        logger.info("📊 Chạy Performance Testing...")
        results = await self.performance_tester.run_all_tests()
        await self.save_results({'performance': results}, 'performance_only')
        return results
    
    async def run_security_only(self) -> Dict[str, Any]:
        """Chỉ chạy security analysis"""
        logger.info("🔒 Chạy Security Analysis...")
        results = await self.security_analyzer.run_all_tests()
        await self.save_results({'security': results}, 'security_only')
        return results
    
    async def run_accuracy_only(self) -> Dict[str, Any]:
        """Chỉ chạy accuracy evaluation"""
        logger.info("🎯 Chạy Accuracy Evaluation...")
        results = await self.accuracy_evaluator.run_all_tests()
        await self.save_results({'accuracy': results}, 'accuracy_only')
        return results
    
    async def save_results(self, results: Dict, filename_prefix: str = "benchmark"):
        """Lưu kết quả benchmark"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = f"results/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 Kết quả đã được lưu: {filepath}")
    
    async def generate_reports(self):
        """Tạo báo cáo tổng hợp"""
        if self.config['output']['generate_html_report']:
            await self.report_generator.generate_html_report(self.results)
        
        if self.config['output']['generate_pdf_report']:
            await self.report_generator.generate_pdf_report(self.results)
    
    def get_summary(self) -> Dict[str, Any]:
        """Lấy tóm tắt kết quả benchmark"""
        if not self.results:
            return {"error": "Chưa có kết quả benchmark"}
        
        summary = {
            "performance_summary": self._summarize_performance(),
            "security_summary": self._summarize_security(),
            "accuracy_summary": self._summarize_accuracy(),
            "overall_score": self._calculate_overall_score()
        }
        
        return summary
    
    def _summarize_performance(self) -> Dict:
        """Tóm tắt kết quả performance"""
        if 'performance' not in self.results:
            return {}
        
        perf = self.results['performance']
        return {
            "avg_tps": perf.get('avg_tps', 0),
            "avg_latency_ms": perf.get('avg_latency_ms', 0),
            "cpu_usage_percent": perf.get('cpu_usage_percent', 0),
            "memory_usage_percent": perf.get('memory_usage_percent', 0),
            "status": "good" if perf.get('avg_tps', 0) > 10 else "needs_improvement"
        }
    
    def _summarize_security(self) -> Dict:
        """Tóm tắt kết quả security"""
        if 'security' not in self.results:
            return {}
        
        sec = self.results['security']
        return {
            "security_score": sec.get('overall_security_score', 0),
            "vulnerabilities_found": len(sec.get('vulnerabilities', [])),
            "consensus_strength": sec.get('consensus_strength', 'unknown'),
            "status": "secure" if sec.get('overall_security_score', 0) > 0.8 else "vulnerable"
        }
    
    def _summarize_accuracy(self) -> Dict:
        """Tóm tắt kết quả accuracy"""
        if 'accuracy' not in self.results:
            return {}
        
        acc = self.results['accuracy']
        return {
            "material_verification_accuracy": acc.get('verification_accuracy', 0),
            "f1_score": acc.get('f1_score', 0),
            "precision": acc.get('precision', 0),
            "recall": acc.get('recall', 0),
            "status": "excellent" if acc.get('f1_score', 0) > 0.9 else "good" if acc.get('f1_score', 0) > 0.7 else "needs_improvement"
        }
    
    def _calculate_overall_score(self) -> float:
        """Tính điểm tổng thể"""
        scores = []
        
        if 'performance' in self.results:
            # Performance score (0-1)
            tps = self.results['performance'].get('avg_tps', 0)
            perf_score = min(tps / 100, 1.0)  # Normalize to max 100 TPS
            scores.append(perf_score * 0.3)  # 30% weight
        
        if 'security' in self.results:
            # Security score (0-1)
            sec_score = self.results['security'].get('overall_security_score', 0)
            scores.append(sec_score * 0.4)  # 40% weight
        
        if 'accuracy' in self.results:
            # Accuracy score (0-1)
            acc_score = self.results['accuracy'].get('f1_score', 0)
            scores.append(acc_score * 0.3)  # 30% weight
        
        return sum(scores) if scores else 0.0

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SpectroChain-Dental Benchmark Suite")
    parser.add_argument("--mode", choices=["full", "performance", "security", "accuracy"], 
                       default="full", help="Chế độ benchmark")
    parser.add_argument("--config", default="benchmark_config.json", 
                       help="File cấu hình benchmark")
    parser.add_argument("--output-dir", default="results", 
                       help="Thư mục lưu kết quả")
    
    args = parser.parse_args()
    
    async def main():
        benchmark = BenchmarkSuite(args.config)
        
        if args.mode == "full":
            results = await benchmark.run_full_benchmark()
        elif args.mode == "performance":
            results = await benchmark.run_performance_only()
        elif args.mode == "security":
            results = await benchmark.run_security_only()
        elif args.mode == "accuracy":
            results = await benchmark.run_accuracy_only()
        
        print("\n📊 TÓNG TẮT KẾT QUẢ BENCHMARK:")
        print("=" * 50)
        summary = benchmark.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
    
    asyncio.run(main())
