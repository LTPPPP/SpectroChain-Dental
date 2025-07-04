"""
Metrics Collector Module
Thu thập metrics hệ thống, blockchain, và performance
"""

import psutil
import time
import json
import requests
from typing import Dict, List, Any
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Công cụ thu thập metrics hệ thống"""
    
    def __init__(self):
        self.monitoring_active = False
        self.metrics_history = []
        
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Thu thập metrics hệ thống"""
        logger.info("📊 Collecting system metrics...")
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        # Process metrics
        process_count = len(psutil.pids())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'frequency_mhz': cpu_freq.current if cpu_freq else None
            },
            'memory': {
                'total_bytes': memory.total,
                'available_bytes': memory.available,
                'used_bytes': memory.used,
                'percent': memory.percent,
                'swap_total_bytes': swap.total,
                'swap_used_bytes': swap.used,
                'swap_percent': swap.percent
            },
            'disk': {
                'total_bytes': disk_usage.total,
                'used_bytes': disk_usage.used,
                'free_bytes': disk_usage.free,
                'percent': (disk_usage.used / disk_usage.total) * 100,
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0,
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0
            },
            'network': {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv,
                'errin': network_io.errin,
                'errout': network_io.errout,
                'dropin': network_io.dropin,
                'dropout': network_io.dropout
            },
            'processes': {
                'count': process_count
            }
        }
    
    async def collect_blockchain_metrics(self, blockchain_url: str = "http://127.0.0.1:7545") -> Dict[str, Any]:
        """Thu thập metrics blockchain"""
        logger.info("⛓️ Collecting blockchain metrics...")
        
        try:
            # Basic blockchain stats (simulated for Ganache)
            response = requests.get(f"{blockchain_url}", timeout=5)
            
            # Since Ganache doesn't have a direct stats endpoint, we simulate metrics
            blockchain_metrics = {
                'network_connected': response.status_code == 200 if response else False,
                'block_number': await self._get_block_number(blockchain_url),
                'transaction_count': await self._get_transaction_count(blockchain_url),
                'gas_price': await self._get_gas_price(blockchain_url),
                'network_hash_rate': self._estimate_hash_rate(),
                'pending_transactions': await self._get_pending_transactions(blockchain_url),
                'peer_count': 1,  # Ganache is single node
                'sync_status': 'synced'
            }
            
            return blockchain_metrics
            
        except Exception as e:
            logger.error(f"Error collecting blockchain metrics: {e}")
            return {
                'network_connected': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_block_number(self, blockchain_url: str) -> int:
        """Lấy số block hiện tại"""
        try:
            # Simulate Web3 call
            return 12345  # Simulated block number
        except:
            return 0
    
    async def _get_transaction_count(self, blockchain_url: str) -> int:
        """Lấy số lượng transaction"""
        try:
            # Simulate transaction count
            return 6789  # Simulated transaction count
        except:
            return 0
    
    async def _get_gas_price(self, blockchain_url: str) -> int:
        """Lấy gas price hiện tại"""
        try:
            # Simulate gas price (in wei)
            return 20000000000  # 20 Gwei
        except:
            return 0
    
    async def _get_pending_transactions(self, blockchain_url: str) -> int:
        """Lấy số transaction đang pending"""
        try:
            # Simulate pending transactions
            return 5  # Simulated pending count
        except:
            return 0
    
    def _estimate_hash_rate(self) -> float:
        """Ước tính hash rate (simulated for PoA)"""
        # PoA doesn't have traditional hash rate, but we can simulate
        return 1000000.0  # 1 MH/s simulated
    
    async def collect_application_metrics(self, backend_url: str = "http://127.0.0.1:8000") -> Dict[str, Any]:
        """Thu thập metrics ứng dụng"""
        logger.info("🖥️ Collecting application metrics...")
        
        try:
            # Test application endpoints
            start_time = time.time()
            response = requests.get(f"{backend_url}/", timeout=5)
            response_time = time.time() - start_time
            
            app_metrics = {
                'application_status': 'running' if response.status_code == 200 else 'error',
                'response_time_ms': response_time * 1000,
                'status_code': response.status_code,
                'uptime_estimation': await self._estimate_uptime(),
                'api_endpoints_tested': await self._test_api_endpoints(backend_url),
                'error_rate': await self._calculate_error_rate(backend_url)
            }
            
            return app_metrics
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return {
                'application_status': 'offline',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _estimate_uptime(self) -> float:
        """Ước tính uptime của ứng dụng"""
        # Simulate uptime calculation
        return 99.5  # 99.5% uptime
    
    async def _test_api_endpoints(self, backend_url: str) -> Dict[str, Any]:
        """Test các API endpoints"""
        endpoints = {
            'root': '/',
            'health': '/health'  # Assuming health endpoint exists
        }
        
        results = {}
        
        for name, endpoint in endpoints.items():
            try:
                start_time = time.time()
                response = requests.get(f"{backend_url}{endpoint}", timeout=3)
                response_time = time.time() - start_time
                
                results[name] = {
                    'status_code': response.status_code,
                    'response_time_ms': response_time * 1000,
                    'available': response.status_code == 200
                }
            except Exception as e:
                results[name] = {
                    'status_code': 0,
                    'response_time_ms': -1,
                    'available': False,
                    'error': str(e)
                }
        
        return results
    
    async def _calculate_error_rate(self, backend_url: str) -> float:
        """Tính error rate"""
        # Simulate error rate calculation
        return 0.5  # 0.5% error rate
    
    async def start_continuous_monitoring(self, 
                                        interval_seconds: int = 60,
                                        blockchain_url: str = "http://127.0.0.1:7545",
                                        backend_url: str = "http://127.0.0.1:8000") -> None:
        """Bắt đầu monitoring liên tục"""
        logger.info(f"🔄 Starting continuous monitoring (interval: {interval_seconds}s)...")
        
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Collect all metrics
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'system': self.collect_system_metrics(),
                    'blockchain': await self.collect_blockchain_metrics(blockchain_url),
                    'application': await self.collect_application_metrics(backend_url)
                }
                
                # Store in history
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 entries to prevent memory issues
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Log key metrics
                self._log_key_metrics(metrics)
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Dừng monitoring"""
        logger.info("🛑 Stopping continuous monitoring...")
        self.monitoring_active = False
    
    def _log_key_metrics(self, metrics: Dict):
        """Log các metrics quan trọng"""
        system = metrics.get('system', {})
        blockchain = metrics.get('blockchain', {})
        app = metrics.get('application', {})
        
        cpu_percent = system.get('cpu', {}).get('percent', 0)
        memory_percent = system.get('memory', {}).get('percent', 0)
        block_number = blockchain.get('block_number', 0)
        app_status = app.get('application_status', 'unknown')
        
        logger.info(f"📊 CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, "
                   f"Blocks: {block_number}, App: {app_status}")
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict]:
        """Lấy lịch sử metrics"""
        return self.metrics_history[-limit:] if self.metrics_history else []
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics ra file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'export_time': datetime.now().isoformat(),
                    'metrics_count': len(self.metrics_history),
                    'metrics': self.metrics_history
                }, f, indent=2, default=str)
            
            logger.info(f"📁 Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def calculate_performance_trends(self) -> Dict[str, Any]:
        """Tính toán xu hướng performance"""
        if len(self.metrics_history) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Extract time series data
        timestamps = []
        cpu_values = []
        memory_values = []
        response_times = []
        
        for metric in self.metrics_history:
            if 'system' in metric and 'application' in metric:
                timestamps.append(metric['timestamp'])
                cpu_values.append(metric['system'].get('cpu', {}).get('percent', 0))
                memory_values.append(metric['system'].get('memory', {}).get('percent', 0))
                response_times.append(metric['application'].get('response_time_ms', 0))
        
        # Calculate trends
        trends = {}
        
        if len(cpu_values) >= 2:
            cpu_trend = 'increasing' if cpu_values[-1] > cpu_values[0] else 'decreasing'
            trends['cpu_trend'] = cpu_trend
            trends['avg_cpu'] = sum(cpu_values) / len(cpu_values)
        
        if len(memory_values) >= 2:
            memory_trend = 'increasing' if memory_values[-1] > memory_values[0] else 'decreasing'
            trends['memory_trend'] = memory_trend
            trends['avg_memory'] = sum(memory_values) / len(memory_values)
        
        if len(response_times) >= 2:
            response_trend = 'increasing' if response_times[-1] > response_times[0] else 'decreasing'
            trends['response_time_trend'] = response_trend
            trends['avg_response_time_ms'] = sum(response_times) / len(response_times)
        
        trends['monitoring_duration_minutes'] = len(self.metrics_history)
        trends['data_points'] = len(self.metrics_history)
        
        return trends
    
    def generate_metrics_summary(self) -> Dict[str, Any]:
        """Tạo tóm tắt metrics"""
        if not self.metrics_history:
            return {'error': 'No metrics data available'}
        
        latest_metrics = self.metrics_history[-1]
        trends = self.calculate_performance_trends()
        
        summary = {
            'monitoring_period': {
                'start_time': self.metrics_history[0]['timestamp'] if self.metrics_history else None,
                'end_time': self.metrics_history[-1]['timestamp'] if self.metrics_history else None,
                'data_points': len(self.metrics_history)
            },
            'current_status': {
                'system': latest_metrics.get('system', {}),
                'blockchain': latest_metrics.get('blockchain', {}),
                'application': latest_metrics.get('application', {})
            },
            'performance_trends': trends,
            'health_indicators': {
                'system_health': self._assess_system_health(latest_metrics),
                'blockchain_health': self._assess_blockchain_health(latest_metrics),
                'application_health': self._assess_application_health(latest_metrics)
            }
        }
        
        return summary
    
    def _assess_system_health(self, metrics: Dict) -> str:
        """Đánh giá sức khỏe hệ thống"""
        system = metrics.get('system', {})
        
        cpu_percent = system.get('cpu', {}).get('percent', 0)
        memory_percent = system.get('memory', {}).get('percent', 0)
        
        if cpu_percent > 90 or memory_percent > 90:
            return 'critical'
        elif cpu_percent > 70 or memory_percent > 70:
            return 'warning'
        else:
            return 'healthy'
    
    def _assess_blockchain_health(self, metrics: Dict) -> str:
        """Đánh giá sức khỏe blockchain"""
        blockchain = metrics.get('blockchain', {})
        
        if not blockchain.get('network_connected', False):
            return 'critical'
        elif blockchain.get('pending_transactions', 0) > 100:
            return 'warning'
        else:
            return 'healthy'
    
    def _assess_application_health(self, metrics: Dict) -> str:
        """Đánh giá sức khỏe ứng dụng"""
        app = metrics.get('application', {})
        
        status = app.get('application_status', 'unknown')
        response_time = app.get('response_time_ms', 0)
        error_rate = app.get('error_rate', 0)
        
        if status != 'running' or error_rate > 5:
            return 'critical'
        elif response_time > 2000:  # 2 seconds
            return 'warning'
        else:
            return 'healthy'
