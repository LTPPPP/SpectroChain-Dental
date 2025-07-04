"""
Report Generator Module
Tạo báo cáo HTML, PDF và visualizations từ kết quả benchmark
"""

import json
import os
from typing import Dict, Any
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Công cụ tạo báo cáo benchmark"""
    
    def __init__(self):
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    async def generate_html_report(self, results: Dict[str, Any]) -> str:
        """Tạo báo cáo HTML"""
        logger.info("📄 Generating HTML report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.output_dir}/benchmark_report_{timestamp}.html"
        
        # Generate visualizations first
        await self._generate_visualizations(results)
        
        html_content = self._create_html_content(results, timestamp)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"📊 HTML report generated: {report_path}")
        return report_path
    
    def _create_html_content(self, results: Dict[str, Any], timestamp: str) -> str:
        """Tạo nội dung HTML"""
        
        # Get summary information
        benchmark_info = results.get('benchmark_info', {})
        performance_summary = results.get('performance', {}).get('summary', {})
        security_summary = results.get('security', {})
        accuracy_summary = results.get('accuracy', {}).get('summary', {})
        
        html = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SpectroChain-Dental Benchmark Report</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 SpectroChain-Dental Benchmark Report</h1>
            <p class="subtitle">Báo cáo đánh giá toàn diện hệ thống</p>
            <div class="info-box">
                <p><strong>Ngày tạo:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                <p><strong>Thời gian benchmark:</strong> {benchmark_info.get('duration_seconds', 0):.2f} giây</p>
            </div>
        </header>

        <div class="summary-cards">
            <div class="card performance">
                <h3>📊 Hiệu Năng</h3>
                <div class="metric">
                    <span class="value">{performance_summary.get('avg_tps', 0):.2f}</span>
                    <span class="unit">TPS</span>
                </div>
                <div class="grade">{performance_summary.get('grade', 'N/A')}</div>
            </div>
            
            <div class="card security">
                <h3>🔒 Bảo Mật</h3>
                <div class="metric">
                    <span class="value">{security_summary.get('overall_security_score', 0):.3f}</span>
                    <span class="unit">Score</span>
                </div>
                <div class="grade">{self._get_security_grade(security_summary.get('overall_security_score', 0))}</div>
            </div>
            
            <div class="card accuracy">
                <h3>🎯 Độ Chính Xác</h3>
                <div class="metric">
                    <span class="value">{accuracy_summary.get('overall_accuracy_score', 0):.3f}</span>
                    <span class="unit">Score</span>
                </div>
                <div class="grade">{accuracy_summary.get('grade', 'N/A')}</div>
            </div>
        </div>

        {self._generate_performance_section(results.get('performance', {}))}
        {self._generate_security_section(results.get('security', {}))}
        {self._generate_accuracy_section(results.get('accuracy', {}))}
        {self._generate_system_metrics_section(results.get('system_metrics', {}))}
        {self._generate_recommendations_section(results)}

        <footer>
            <p>Báo cáo được tạo bởi SpectroChain-Dental Benchmark Suite</p>
            <p>Timestamp: {timestamp}</p>
        </footer>
    </div>

    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
"""
        return html
    
    def _get_css_styles(self) -> str:
        """CSS styles cho HTML report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            margin-top: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .info-box {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }

        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .card {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .card h3 {
            margin-bottom: 20px;
            font-size: 1.3em;
        }

        .metric {
            margin-bottom: 15px;
        }

        .value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }

        .unit {
            font-size: 1em;
            color: #666;
            margin-left: 5px;
        }

        .grade {
            font-size: 1.5em;
            font-weight: bold;
            padding: 10px;
            border-radius: 50px;
            background: #28a745;
            color: white;
        }

        .section {
            margin-bottom: 40px;
            background: #f8f9fa;
            padding: 30px;
            border-radius: 10px;
        }

        .section h2 {
            margin-bottom: 20px;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .metric-item {
            background: white;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }

        .metric-label {
            font-weight: bold;
            color: #666;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 1.3em;
            color: #333;
        }

        .chart-container {
            margin: 20px 0;
            text-align: center;
        }

        .recommendations {
            background: #e8f4f8;
            border-left: 4px solid #17a2b8;
        }

        .vulnerability {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            margin: 10px 0;
            padding: 15px;
            border-radius: 5px;
        }

        .vulnerability h4 {
            color: #721c24;
            margin-bottom: 10px;
        }

        footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            color: #666;
        }

        @media (max-width: 768px) {
            .container {
                margin: 10px;
                padding: 10px;
            }
            
            .summary-cards {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _generate_performance_section(self, performance: Dict) -> str:
        """Tạo section hiệu năng"""
        if not performance:
            return ""
        
        summary = performance.get('summary', {})
        tps_tests = performance.get('tps_tests', {})
        latency_tests = performance.get('latency_tests', {})
        resource_usage = performance.get('resource_usage', {})
        
        return f"""
        <div class="section">
            <h2>📊 Hiệu Năng (Performance)</h2>
            
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-label">Thông lượng (TPS)</div>
                    <div class="metric-value">{tps_tests.get('avg_tps', 0):.2f} TPS</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Độ trễ trung bình</div>
                    <div class="metric-value">{latency_tests.get('avg_latency_ms', 0):.2f} ms</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Sử dụng CPU</div>
                    <div class="metric-value">{resource_usage.get('avg_cpu_percent', 0):.1f}%</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Sử dụng RAM</div>
                    <div class="metric-value">{resource_usage.get('avg_memory_percent', 0):.1f}%</div>
                </div>
            </div>
            
            <h3>Chi tiết TPS theo số user:</h3>
            <div class="metrics-grid">
                {self._generate_tps_details(tps_tests)}
            </div>
            
            <h3>Phân tích Latency:</h3>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-label">P95 Latency</div>
                    <div class="metric-value">{latency_tests.get('p95_latency_ms', 0):.2f} ms</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">P99 Latency</div>
                    <div class="metric-value">{latency_tests.get('p99_latency_ms', 0):.2f} ms</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Max Latency</div>
                    <div class="metric-value">{latency_tests.get('max_latency_ms', 0):.2f} ms</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_tps_details(self, tps_tests: Dict) -> str:
        """Tạo chi tiết TPS"""
        details = ""
        for key, value in tps_tests.items():
            if key.startswith('tps_') and key.endswith('_users'):
                users = key.replace('tps_', '').replace('_users', '')
                details += f"""
                <div class="metric-item">
                    <div class="metric-label">{users} Users</div>
                    <div class="metric-value">{value:.2f} TPS</div>
                </div>
                """
        return details
    
    def _generate_security_section(self, security: Dict) -> str:
        """Tạo section bảo mật"""
        if not security:
            return ""
        
        overall_score = security.get('overall_security_score', 0)
        vulnerabilities = security.get('vulnerabilities', [])
        stride_analysis = security.get('stride_analysis', {})
        
        vuln_html = ""
        for vuln in vulnerabilities:
            vuln_html += f"""
            <div class="vulnerability">
                <h4>{vuln.get('threat', 'Unknown Threat')} - {vuln.get('severity', 'Unknown').upper()}</h4>
                <p>Điểm số: {vuln.get('score', 0):.3f}</p>
                <p>Khuyến nghị: {', '.join(vuln.get('recommendations', []))}</p>
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>🔒 Bảo Mật (Security)</h2>
            
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-label">Điểm bảo mật tổng thể</div>
                    <div class="metric-value">{overall_score:.3f}</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Số lỗ hổng phát hiện</div>
                    <div class="metric-value">{len(vulnerabilities)}</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">STRIDE Score</div>
                    <div class="metric-value">{stride_analysis.get('overall_stride_score', 0):.3f}</div>
                </div>
            </div>
            
            <h3>Lỗ hổng bảo mật:</h3>
            {vuln_html if vulnerabilities else '<p>Không phát hiện lỗ hổng nghiêm trọng.</p>'}
            
            <h3>Phân tích STRIDE:</h3>
            <div class="metrics-grid">
                {self._generate_stride_details(stride_analysis)}
            </div>
        </div>
        """
    
    def _generate_stride_details(self, stride_analysis: Dict) -> str:
        """Tạo chi tiết STRIDE"""
        details = ""
        stride_threats = ['Spoofing', 'Tampering', 'Repudiation', 'Information_Disclosure', 'Denial_of_Service', 'Elevation_of_Privilege']
        
        for threat in stride_threats:
            if threat in stride_analysis:
                analysis = stride_analysis[threat]
                vuln_score = analysis.get('vulnerability_score', 0)
                security_score = 1 - vuln_score
                
                details += f"""
                <div class="metric-item">
                    <div class="metric-label">{threat.replace('_', ' ')}</div>
                    <div class="metric-value">{security_score:.3f}</div>
                </div>
                """
        
        return details
    
    def _generate_accuracy_section(self, accuracy: Dict) -> str:
        """Tạo section độ chính xác"""
        if not accuracy:
            return ""
        
        summary = accuracy.get('summary', {})
        material_verification = accuracy.get('material_verification', {})
        ml_evaluation = accuracy.get('ml_model_evaluation', {})
        
        return f"""
        <div class="section">
            <h2>🎯 Độ Chính Xác (Accuracy)</h2>
            
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-label">Độ chính xác xác minh vật liệu</div>
                    <div class="metric-value">{material_verification.get('overall_verification_accuracy', 0):.3f}</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Tỷ lệ phát hiện hàng giả</div>
                    <div class="metric-value">{material_verification.get('fake_detection_rate', 0):.3f}</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Best ML Model</div>
                    <div class="metric-value">{summary.get('best_ml_model', 'N/A')}</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Best F1 Score</div>
                    <div class="metric-value">{summary.get('best_ml_f1', 0):.3f}</div>
                </div>
            </div>
            
            <h3>Chi tiết xác minh vật liệu:</h3>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-label">Tỷ lệ xác minh đúng</div>
                    <div class="metric-value">{material_verification.get('genuine_verification_rate', 0):.3f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">False Positive Rate</div>
                    <div class="metric-value">{material_verification.get('false_positive_rate', 0):.3f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">False Negative Rate</div>
                    <div class="metric-value">{material_verification.get('false_negative_rate', 0):.3f}</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_system_metrics_section(self, system_metrics: Dict) -> str:
        """Tạo section system metrics"""
        if not system_metrics:
            return ""
        
        return f"""
        <div class="section">
            <h2>🖥️ System Metrics</h2>
            
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-label">CPU Cores</div>
                    <div class="metric-value">{system_metrics.get('cpu', {}).get('count', 'N/A')}</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Total Memory</div>
                    <div class="metric-value">{self._bytes_to_gb(system_metrics.get('memory', {}).get('total_bytes', 0)):.1f} GB</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Disk Usage</div>
                    <div class="metric-value">{system_metrics.get('disk', {}).get('percent', 0):.1f}%</div>
                </div>
                
                <div class="metric-item">
                    <div class="metric-label">Process Count</div>
                    <div class="metric-value">{system_metrics.get('processes', {}).get('count', 'N/A')}</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_recommendations_section(self, results: Dict) -> str:
        """Tạo section khuyến nghị"""
        recommendations = []
        
        # Performance recommendations
        performance = results.get('performance', {})
        if performance.get('summary', {}).get('performance_score', 0) < 0.7:
            recommendations.append("Cải thiện hiệu năng: Tối ưu hóa database queries và caching")
        
        # Security recommendations
        security = results.get('security', {})
        if security.get('overall_security_score', 0) < 0.8:
            recommendations.extend(security.get('recommendations', []))
        
        # Accuracy recommendations
        accuracy = results.get('accuracy', {})
        if accuracy.get('summary', {}).get('overall_accuracy_score', 0) < 0.9:
            recommendations.append("Cải thiện độ chính xác: Thu thập thêm dữ liệu training và fine-tuning models")
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Hệ thống đang hoạt động tốt",
                "Tiếp tục monitoring và bảo trì định kỳ",
                "Cập nhật security patches thường xuyên"
            ]
        
        rec_html = ""
        for i, rec in enumerate(recommendations, 1):
            rec_html += f"<p>{i}. {rec}</p>"
        
        return f"""
        <div class="section recommendations">
            <h2>💡 Khuyến Nghị</h2>
            {rec_html}
        </div>
        """
    
    def _get_security_grade(self, score: float) -> str:
        """Lấy grade bảo mật"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        else:
            return "D"
    
    def _bytes_to_gb(self, bytes_val: int) -> float:
        """Convert bytes to GB"""
        return bytes_val / (1024**3) if bytes_val > 0 else 0
    
    def _get_javascript(self) -> str:
        """JavaScript cho interactive features"""
        return """
        // Add some interactivity
        document.addEventListener('DOMContentLoaded', function() {
            // Animate cards on scroll
            const cards = document.querySelectorAll('.card');
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }
                });
            });
            
            cards.forEach(card => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                card.style.transition = 'all 0.5s ease';
                observer.observe(card);
            });
        });
        """
    
    async def generate_pdf_report(self, results: Dict[str, Any]) -> str:
        """Tạo báo cáo PDF (simplified)"""
        logger.info("📄 PDF generation would require additional libraries (weasyprint, reportlab)")
        logger.info("For now, please use the HTML report and print to PDF from browser")
        return "PDF generation not implemented - use HTML report"
    
    async def _generate_visualizations(self, results: Dict[str, Any]):
        """Tạo các visualizations"""
        logger.info("📊 Generating visualizations...")
        
        try:
            # Performance charts
            await self._create_performance_charts(results.get('performance', {}))
            
            # Security charts
            await self._create_security_charts(results.get('security', {}))
            
            # Accuracy charts
            await self._create_accuracy_charts(results.get('accuracy', {}))
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    async def _create_performance_charts(self, performance: Dict):
        """Tạo performance charts"""
        if not performance:
            return
        
        # TPS Chart
        tps_data = performance.get('tps_tests', {})
        if tps_data:
            users = []
            tps_values = []
            
            for key, value in tps_data.items():
                if key.startswith('tps_') and key.endswith('_users'):
                    user_count = key.replace('tps_', '').replace('_users', '')
                    users.append(int(user_count))
                    tps_values.append(value)
            
            if users and tps_values:
                plt.figure(figsize=(10, 6))
                plt.plot(users, tps_values, marker='o', linewidth=2, markersize=8)
                plt.title('Transactions Per Second vs Concurrent Users', fontsize=14, fontweight='bold')
                plt.xlabel('Concurrent Users')
                plt.ylabel('TPS')
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{self.output_dir}/tps_chart.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    async def _create_security_charts(self, security: Dict):
        """Tạo security charts"""
        if not security:
            return
        
        # STRIDE Analysis Chart
        stride_analysis = security.get('stride_analysis', {})
        if stride_analysis:
            threats = []
            scores = []
            
            stride_threats = ['Spoofing', 'Tampering', 'Repudiation', 'Information_Disclosure', 'Denial_of_Service', 'Elevation_of_Privilege']
            
            for threat in stride_threats:
                if threat in stride_analysis:
                    threats.append(threat.replace('_', ' '))
                    vuln_score = stride_analysis[threat].get('vulnerability_score', 0)
                    security_score = 1 - vuln_score
                    scores.append(security_score)
            
            if threats and scores:
                plt.figure(figsize=(12, 6))
                bars = plt.bar(threats, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
                plt.title('STRIDE Security Analysis', fontsize=14, fontweight='bold')
                plt.ylabel('Security Score')
                plt.ylim(0, 1)
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, value in zip(bars, scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/stride_chart.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    async def _create_accuracy_charts(self, accuracy: Dict):
        """Tạo accuracy charts"""
        if not accuracy:
            return
        
        # Model comparison chart
        ml_evaluation = accuracy.get('ml_model_evaluation', {})
        if ml_evaluation:
            models = []
            accuracies = []
            f1_scores = []
            
            for model_name, results in ml_evaluation.items():
                if isinstance(results, dict) and 'accuracy' in results:
                    models.append(model_name)
                    accuracies.append(results['accuracy'])
                    f1_scores.append(results['f1_score'])
            
            if models:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Accuracy comparison
                ax1.bar(models, accuracies, color='skyblue')
                ax1.set_title('Model Accuracy Comparison')
                ax1.set_ylabel('Accuracy')
                ax1.set_ylim(0, 1)
                
                # F1 Score comparison
                ax2.bar(models, f1_scores, color='lightcoral')
                ax2.set_title('Model F1 Score Comparison')
                ax2.set_ylabel('F1 Score')
                ax2.set_ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    async def generate_dashboard(self, results: Dict[str, Any]) -> str:
        """Tạo interactive dashboard với Plotly"""
        logger.info("📊 Generating interactive dashboard...")
        
        try:
            # Create dashboard with multiple charts
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('TPS vs Users', 'STRIDE Analysis', 'Model Accuracy', 'System Resource Usage'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )
            
            # Add TPS chart
            performance = results.get('performance', {})
            tps_data = performance.get('tps_tests', {})
            if tps_data:
                users = []
                tps_values = []
                for key, value in tps_data.items():
                    if key.startswith('tps_') and key.endswith('_users'):
                        user_count = key.replace('tps_', '').replace('_users', '')
                        users.append(int(user_count))
                        tps_values.append(value)
                
                if users:
                    fig.add_trace(
                        go.Scatter(x=users, y=tps_values, mode='lines+markers', name='TPS'),
                        row=1, col=1
                    )
            
            # Add STRIDE chart
            security = results.get('security', {})
            stride_analysis = security.get('stride_analysis', {})
            if stride_analysis:
                threats = []
                scores = []
                stride_threats = ['Spoofing', 'Tampering', 'Repudiation', 'Information_Disclosure', 'Denial_of_Service', 'Elevation_of_Privilege']
                
                for threat in stride_threats:
                    if threat in stride_analysis:
                        threats.append(threat.replace('_', ' '))
                        vuln_score = stride_analysis[threat].get('vulnerability_score', 0)
                        security_score = 1 - vuln_score
                        scores.append(security_score)
                
                if threats:
                    fig.add_trace(
                        go.Bar(x=threats, y=scores, name='Security Score'),
                        row=1, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title="SpectroChain-Dental Benchmark Dashboard",
                height=800,
                showlegend=False
            )
            
            # Save dashboard
            dashboard_path = f"{self.output_dir}/dashboard.html"
            fig.write_html(dashboard_path)
            
            logger.info(f"📊 Interactive dashboard created: {dashboard_path}")
            return dashboard_path
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return ""
