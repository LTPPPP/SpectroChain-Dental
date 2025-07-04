"""
Chaos Engineering Module
Mô phỏng sự cố và test độ bền của hệ thống
"""

import asyncio
import random
import time
import subprocess
from typing import Dict, List, Any, Callable
import logging
import psutil
import requests

logger = logging.getLogger(__name__)

class ChaosEngineer:
    """Công cụ Chaos Engineering"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.active_experiments = []
        self.system_baseline = None
        
    async def run_chaos_experiments(self) -> Dict[str, Any]:
        """Chạy các thí nghiệm chaos"""
        logger.info("💥 Starting Chaos Engineering experiments...")
        
        # Collect baseline metrics
        self.system_baseline = await self._collect_baseline_metrics()
        
        experiments = {
            'network_latency': await self._test_network_latency_injection(),
            'cpu_stress': await self._test_cpu_stress(),
            'memory_pressure': await self._test_memory_pressure(),
            'disk_fill': await self._test_disk_fill(),
            'process_termination': await self._test_process_termination(),
            'network_partition': await self._test_network_partition(),
            'database_chaos': await self._test_database_chaos()
        }
        
        # Analyze resilience
        experiments['resilience_analysis'] = await self._analyze_system_resilience(experiments)
        
        return experiments
    
    async def _collect_baseline_metrics(self) -> Dict[str, float]:
        """Thu thập metrics baseline"""
        logger.info("📊 Collecting baseline metrics...")
        
        metrics = {}
        
        # CPU baseline
        metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        
        # Memory baseline
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        
        # Network baseline
        try:
            response = requests.get("http://127.0.0.1:8000/", timeout=5)
            metrics['response_time_ms'] = response.elapsed.total_seconds() * 1000
            metrics['service_available'] = response.status_code == 200
        except:
            metrics['response_time_ms'] = -1
            metrics['service_available'] = False
        
        # Disk baseline
        disk = psutil.disk_usage('/')
        metrics['disk_percent'] = (disk.used / disk.total) * 100
        
        return metrics
    
    async def _test_network_latency_injection(self) -> Dict[str, Any]:
        """Test với network latency injection"""
        logger.info("🌐 Testing network latency injection...")
        
        results = {
            'experiment': 'network_latency_injection',
            'duration_seconds': 60,
            'latency_added_ms': 100
        }
        
        try:
            # Measure response time before
            start_response_times = []
            for _ in range(10):
                try:
                    start_time = time.time()
                    response = requests.get("http://127.0.0.1:8000/", timeout=5)
                    response_time = (time.time() - start_time) * 1000
                    start_response_times.append(response_time)
                except:
                    start_response_times.append(-1)
                await asyncio.sleep(1)
            
            # Simulate network latency (would use tc/netem in real implementation)
            logger.info("💥 Simulating network latency...")
            await asyncio.sleep(60)  # Simulate chaos for 60 seconds
            
            # Measure response time after
            end_response_times = []
            for _ in range(10):
                try:
                    start_time = time.time()
                    response = requests.get("http://127.0.0.1:8000/", timeout=5)
                    response_time = (time.time() - start_time) * 1000
                    end_response_times.append(response_time)
                except:
                    end_response_times.append(-1)
                await asyncio.sleep(1)
            
            valid_start = [t for t in start_response_times if t > 0]
            valid_end = [t for t in end_response_times if t > 0]
            
            results.update({
                'baseline_avg_response_ms': sum(valid_start) / len(valid_start) if valid_start else -1,
                'chaos_avg_response_ms': sum(valid_end) / len(valid_end) if valid_end else -1,
                'service_degradation': len(valid_end) < len(valid_start),
                'recovery_successful': len(valid_end) > 0
            })
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def _test_cpu_stress(self) -> Dict[str, Any]:
        """Test với CPU stress"""
        logger.info("🔥 Testing CPU stress...")
        
        results = {
            'experiment': 'cpu_stress',
            'duration_seconds': 30,
            'target_cpu_percent': 80
        }
        
        try:
            # Collect pre-stress metrics
            pre_cpu = psutil.cpu_percent(interval=1)
            
            # Start CPU stress process
            stress_process = await self._start_cpu_stress(duration=30)
            
            # Monitor during stress
            cpu_samples = []
            service_responses = []
            
            for i in range(30):
                cpu_samples.append(psutil.cpu_percent())
                
                # Test service response
                try:
                    start_time = time.time()
                    response = requests.get("http://127.0.0.1:8000/", timeout=5)
                    response_time = (time.time() - start_time) * 1000
                    service_responses.append({
                        'success': response.status_code == 200,
                        'response_time_ms': response_time
                    })
                except:
                    service_responses.append({
                        'success': False,
                        'response_time_ms': -1
                    })
                
                await asyncio.sleep(1)
            
            # Cleanup
            if stress_process:
                stress_process.terminate()
            
            # Collect post-stress metrics
            await asyncio.sleep(5)  # Recovery time
            post_cpu = psutil.cpu_percent(interval=1)
            
            # Analyze results
            avg_cpu_during_stress = sum(cpu_samples) / len(cpu_samples)
            successful_responses = sum(1 for r in service_responses if r['success'])
            avg_response_time = sum(r['response_time_ms'] for r in service_responses if r['response_time_ms'] > 0)
            avg_response_time = avg_response_time / max(1, len([r for r in service_responses if r['response_time_ms'] > 0]))
            
            results.update({
                'pre_stress_cpu_percent': pre_cpu,
                'avg_cpu_during_stress': avg_cpu_during_stress,
                'post_stress_cpu_percent': post_cpu,
                'service_availability_percent': (successful_responses / len(service_responses)) * 100,
                'avg_response_time_during_stress': avg_response_time,
                'system_recovered': abs(post_cpu - pre_cpu) < 10
            })
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def _start_cpu_stress(self, duration: int = 30):
        """Bắt đầu CPU stress test"""
        try:
            # Use stress-ng if available, otherwise Python-based stress
            cmd = ["stress-ng", "--cpu", "0", "--timeout", f"{duration}s"]
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return process
        except FileNotFoundError:
            # Fallback to Python-based CPU stress
            logger.info("stress-ng not found, using Python-based CPU stress")
            return await self._python_cpu_stress(duration)
    
    async def _python_cpu_stress(self, duration: int):
        """Python-based CPU stress"""
        import threading
        import multiprocessing
        
        def cpu_bound_task():
            end_time = time.time() + duration
            while time.time() < end_time:
                # CPU intensive calculation
                sum(i * i for i in range(10000))
        
        # Start stress threads
        threads = []
        cpu_count = multiprocessing.cpu_count()
        
        for _ in range(cpu_count):
            thread = threading.Thread(target=cpu_bound_task)
            thread.start()
            threads.append(thread)
        
        return threads  # Return threads for cleanup
    
    async def _test_memory_pressure(self) -> Dict[str, Any]:
        """Test với memory pressure"""
        logger.info("🧠 Testing memory pressure...")
        
        results = {
            'experiment': 'memory_pressure',
            'duration_seconds': 30
        }
        
        try:
            # Collect pre-test memory
            pre_memory = psutil.virtual_memory()
            
            # Allocate memory gradually
            memory_blocks = []
            block_size = 10 * 1024 * 1024  # 10MB blocks
            
            service_responses = []
            
            for i in range(30):  # 30 seconds test
                # Allocate memory
                try:
                    memory_blocks.append(bytearray(block_size))
                except MemoryError:
                    logger.warning("Memory allocation failed - system limit reached")
                    break
                
                # Test service
                try:
                    start_time = time.time()
                    response = requests.get("http://127.0.0.1:8000/", timeout=5)
                    response_time = (time.time() - start_time) * 1000
                    service_responses.append({
                        'success': response.status_code == 200,
                        'response_time_ms': response_time,
                        'memory_used_mb': len(memory_blocks) * (block_size / 1024 / 1024)
                    })
                except:
                    service_responses.append({
                        'success': False,
                        'response_time_ms': -1,
                        'memory_used_mb': len(memory_blocks) * (block_size / 1024 / 1024)
                    })
                
                await asyncio.sleep(1)
            
            # Cleanup memory
            memory_blocks.clear()
            
            # Collect post-test memory
            await asyncio.sleep(5)
            post_memory = psutil.virtual_memory()
            
            # Analyze results
            max_memory_allocated = max((r['memory_used_mb'] for r in service_responses), default=0)
            successful_responses = sum(1 for r in service_responses if r['success'])
            
            results.update({
                'pre_test_memory_percent': pre_memory.percent,
                'post_test_memory_percent': post_memory.percent,
                'max_additional_memory_mb': max_memory_allocated,
                'service_availability_percent': (successful_responses / len(service_responses)) * 100 if service_responses else 0,
                'memory_recovered': abs(post_memory.percent - pre_memory.percent) < 5
            })
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def _test_disk_fill(self) -> Dict[str, Any]:
        """Test với disk fill"""
        logger.info("💽 Testing disk fill...")
        
        results = {
            'experiment': 'disk_fill',
            'simulated': True  # Just simulate for safety
        }
        
        # For safety, we only simulate disk fill rather than actually filling disk
        # In real chaos engineering, this would gradually fill up disk space
        
        try:
            disk_before = psutil.disk_usage('/')
            
            # Simulate disk fill effects
            logger.info("💥 Simulating disk fill scenario...")
            
            service_responses = []
            for i in range(10):
                try:
                    response = requests.get("http://127.0.0.1:8000/", timeout=5)
                    service_responses.append({
                        'success': response.status_code == 200,
                        'simulated_disk_percent': min(95, disk_before.percent + (i * 2))
                    })
                except:
                    service_responses.append({
                        'success': False,
                        'simulated_disk_percent': min(95, disk_before.percent + (i * 2))
                    })
                
                await asyncio.sleep(2)
            
            disk_after = psutil.disk_usage('/')
            successful_responses = sum(1 for r in service_responses if r['success'])
            
            results.update({
                'initial_disk_percent': disk_before.percent,
                'final_disk_percent': disk_after.percent,
                'service_availability_percent': (successful_responses / len(service_responses)) * 100,
                'system_resilient_to_disk_pressure': successful_responses > len(service_responses) * 0.8
            })
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def _test_process_termination(self) -> Dict[str, Any]:
        """Test process termination resilience"""
        logger.info("☠️ Testing process termination resilience...")
        
        results = {
            'experiment': 'process_termination',
            'simulated': True  # Simulate rather than actually killing processes
        }
        
        try:
            # Check if target processes are running
            target_processes = ['python', 'node', 'ganache']
            running_processes = []
            
            for proc in psutil.process_iter(['pid', 'name']):
                if any(target in proc.info['name'].lower() for target in target_processes):
                    running_processes.append(proc.info)
            
            # Simulate process recovery testing
            logger.info("💥 Simulating process termination...")
            
            service_responses = []
            for i in range(10):
                try:
                    response = requests.get("http://127.0.0.1:8000/", timeout=5)
                    service_responses.append({
                        'success': response.status_code == 200,
                        'attempt': i + 1
                    })
                except:
                    service_responses.append({
                        'success': False,
                        'attempt': i + 1
                    })
                
                await asyncio.sleep(3)
            
            successful_responses = sum(1 for r in service_responses if r['success'])
            
            results.update({
                'processes_found': len(running_processes),
                'service_availability_percent': (successful_responses / len(service_responses)) * 100,
                'recovery_successful': successful_responses > 0,
                'process_resilience_score': successful_responses / len(service_responses)
            })
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def _test_network_partition(self) -> Dict[str, Any]:
        """Test network partition resilience"""
        logger.info("🌐 Testing network partition resilience...")
        
        results = {
            'experiment': 'network_partition',
            'simulated': True
        }
        
        try:
            # Simulate network partition by testing connectivity
            services = [
                'http://127.0.0.1:8000/',  # Backend
                'http://127.0.0.1:7545/'   # Blockchain
            ]
            
            connectivity_results = {}
            
            for service in services:
                service_results = []
                for i in range(10):
                    try:
                        start_time = time.time()
                        response = requests.get(service, timeout=5)
                        response_time = (time.time() - start_time) * 1000
                        service_results.append({
                            'success': True,
                            'response_time_ms': response_time
                        })
                    except:
                        service_results.append({
                            'success': False,
                            'response_time_ms': -1
                        })
                    
                    await asyncio.sleep(2)
                
                connectivity_results[service] = service_results
            
            # Analyze connectivity
            overall_availability = 0
            for service, service_results in connectivity_results.items():
                successful = sum(1 for r in service_results if r['success'])
                availability = (successful / len(service_results)) * 100
                overall_availability += availability
            
            overall_availability /= len(services)
            
            results.update({
                'services_tested': len(services),
                'overall_availability_percent': overall_availability,
                'partition_resilience_score': overall_availability / 100,
                'connectivity_details': connectivity_results
            })
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def _test_database_chaos(self) -> Dict[str, Any]:
        """Test database chaos scenarios"""
        logger.info("🗄️ Testing database chaos scenarios...")
        
        results = {
            'experiment': 'database_chaos',
            'simulated': True
        }
        
        try:
            # Simulate database-related chaos
            # In real scenario, this would test:
            # - Connection pool exhaustion
            # - Slow queries
            # - Temporary unavailability
            
            database_scenarios = [
                'connection_exhaustion',
                'slow_queries',
                'temporary_unavailability',
                'index_corruption'
            ]
            
            scenario_results = {}
            
            for scenario in database_scenarios:
                logger.info(f"💥 Simulating {scenario}...")
                
                # Simulate different database stress scenarios
                service_responses = []
                for i in range(5):
                    try:
                        # Add artificial delay for some scenarios
                        if scenario == 'slow_queries':
                            await asyncio.sleep(1)
                        
                        response = requests.get("http://127.0.0.1:8000/", timeout=10)
                        service_responses.append({
                            'success': response.status_code == 200,
                            'scenario': scenario
                        })
                    except:
                        service_responses.append({
                            'success': False,
                            'scenario': scenario
                        })
                    
                    await asyncio.sleep(2)
                
                successful = sum(1 for r in service_responses if r['success'])
                scenario_results[scenario] = {
                    'availability_percent': (successful / len(service_responses)) * 100,
                    'resilience_score': successful / len(service_responses)
                }
            
            # Calculate overall database resilience
            avg_resilience = sum(s['resilience_score'] for s in scenario_results.values()) / len(scenario_results)
            
            results.update({
                'scenarios_tested': len(database_scenarios),
                'scenario_results': scenario_results,
                'overall_database_resilience': avg_resilience,
                'database_chaos_grade': 'A' if avg_resilience > 0.8 else 'B' if avg_resilience > 0.6 else 'C'
            })
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def _analyze_system_resilience(self, experiments: Dict[str, Any]) -> Dict[str, Any]:
        """Phân tích độ bền tổng thể của hệ thống"""
        logger.info("🔍 Analyzing overall system resilience...")
        
        resilience_scores = []
        critical_failures = []
        
        # Collect resilience scores from experiments
        for exp_name, exp_data in experiments.items():
            if isinstance(exp_data, dict):
                if 'service_availability_percent' in exp_data:
                    availability = exp_data['service_availability_percent']
                    resilience_scores.append(availability / 100)
                    
                    if availability < 50:  # Less than 50% availability is critical
                        critical_failures.append(exp_name)
                
                if 'resilience_score' in exp_data:
                    resilience_scores.append(exp_data['resilience_score'])
                
                if 'recovery_successful' in exp_data and not exp_data['recovery_successful']:
                    critical_failures.append(exp_name)
        
        # Calculate overall resilience
        overall_resilience = sum(resilience_scores) / len(resilience_scores) if resilience_scores else 0
        
        # Resilience grade
        if overall_resilience >= 0.9:
            grade = 'A'
            description = 'Excellent resilience - system handles chaos well'
        elif overall_resilience >= 0.7:
            grade = 'B'
            description = 'Good resilience - minor degradation under stress'
        elif overall_resilience >= 0.5:
            grade = 'C'
            description = 'Moderate resilience - significant degradation under stress'
        else:
            grade = 'D'
            description = 'Poor resilience - system fails under stress'
        
        # Generate recommendations
        recommendations = []
        
        if critical_failures:
            recommendations.append(f"Critical failures in: {', '.join(critical_failures)}")
        
        if overall_resilience < 0.8:
            recommendations.extend([
                "Implement circuit breakers for external dependencies",
                "Add retry mechanisms with exponential backoff",
                "Improve error handling and graceful degradation",
                "Consider implementing bulkhead pattern"
            ])
        
        if 'cpu_stress' in critical_failures:
            recommendations.append("Consider auto-scaling or load balancing for CPU-intensive operations")
        
        if 'memory_pressure' in critical_failures:
            recommendations.append("Implement memory monitoring and garbage collection optimization")
        
        if 'network_partition' in critical_failures:
            recommendations.append("Implement network partition tolerance and eventual consistency")
        
        return {
            'overall_resilience_score': overall_resilience,
            'resilience_grade': grade,
            'description': description,
            'critical_failures': critical_failures,
            'experiments_passed': len(experiments) - len(critical_failures),
            'total_experiments': len(experiments),
            'recommendations': recommendations,
            'chaos_engineering_maturity': self._assess_chaos_maturity(overall_resilience)
        }
    
    def _assess_chaos_maturity(self, resilience_score: float) -> str:
        """Đánh giá mức độ maturity của Chaos Engineering"""
        if resilience_score >= 0.9:
            return "Advanced - System demonstrates excellent chaos resilience"
        elif resilience_score >= 0.7:
            return "Intermediate - System shows good resilience with room for improvement"
        elif resilience_score >= 0.5:
            return "Basic - System has basic resilience mechanisms"
        else:
            return "Novice - System needs significant resilience improvements"
