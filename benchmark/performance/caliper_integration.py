"""
Hyperledger Caliper Integration
Tích hợp với Hyperledger Caliper để benchmark blockchain performance
"""

import json
import os
import subprocess
import yaml
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class CaliperIntegration:
    """Tích hợp Hyperledger Caliper cho blockchain benchmarking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.caliper_workspace = "caliper_workspace"
        self.network_config = None
        self.benchmark_config = None
        
    async def setup_caliper_workspace(self):
        """Thiết lập workspace cho Caliper"""
        logger.info("🔧 Setting up Caliper workspace...")
        
        # Create workspace directory
        os.makedirs(self.caliper_workspace, exist_ok=True)
        
        # Generate network configuration
        await self._generate_network_config()
        
        # Generate benchmark configuration
        await self._generate_benchmark_config()
        
        # Generate smart contract bindings
        await self._generate_contract_bindings()
        
        logger.info("✅ Caliper workspace setup completed")
    
    async def _generate_network_config(self):
        """Tạo network configuration cho Caliper"""
        
        # For Ganache/Ethereum network
        network_config = {
            "caliper": {
                "blockchain": "ethereum",
                "command": {
                    "start": "docker-compose -f docker-compose-ganache.yaml up -d",
                    "end": "docker-compose -f docker-compose-ganache.yaml down"
                }
            },
            "ethereum": {
                "url": self.config.get('blockchain_url', 'http://127.0.0.1:7545'),
                "contractDeployerAddress": "0x...",  # First Ganache account
                "contractDeployerAddressPrivateKey": "0x...",  # Private key
                "fromAddressSeed": "0x...",
                "contracts": {
                    "SpectroChain": {
                        "path": "../blockchain/build/contracts/SpectroChain.json",
                        "gas": 3000000,
                        "gasPrice": 20000000000
                    }
                }
            }
        }
        
        config_path = os.path.join(self.caliper_workspace, "network-config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(network_config, f, default_flow_style=False)
        
        self.network_config = config_path
        logger.info(f"📄 Network config generated: {config_path}")
    
    async def _generate_benchmark_config(self):
        """Tạo benchmark configuration"""
        
        benchmark_config = {
            "test": {
                "name": "SpectroChain Performance Benchmark",
                "description": "Comprehensive performance testing for SpectroChain-Dental",
                "workers": {
                    "type": "local",
                    "number": 4
                },
                "rounds": [
                    {
                        "label": "Material Registration",
                        "description": "Test material registration performance",
                        "txNumber": 1000,
                        "rateControl": {
                            "type": "fixed-rate",
                            "opts": {
                                "tps": 10
                            }
                        },
                        "workload": {
                            "module": "./workloads/register-material.js"
                        }
                    },
                    {
                        "label": "Material Verification",
                        "description": "Test material verification performance",
                        "txNumber": 1000,
                        "rateControl": {
                            "type": "fixed-rate",
                            "opts": {
                                "tps": 20
                            }
                        },
                        "workload": {
                            "module": "./workloads/verify-material.js"
                        }
                    },
                    {
                        "label": "Mixed Workload",
                        "description": "Mixed registration and verification",
                        "txNumber": 2000,
                        "rateControl": {
                            "type": "linear-rate",
                            "opts": {
                                "startingTps": 5,
                                "finishingTps": 50
                            }
                        },
                        "workload": {
                            "module": "./workloads/mixed-operations.js"
                        }
                    }
                ]
            },
            "monitors": [
                {
                    "module": "prometheus",
                    "options": {
                        "url": "http://localhost:9090",
                        "metrics": {
                            "include": ["all"],
                            "queries": [
                                {
                                    "name": "Memory (MB)",
                                    "query": "sum(container_memory_rss{name=~\".*caliper.*\"}) by (name)",
                                    "step": 1,
                                    "label": "name",
                                    "statistic": "avg"
                                },
                                {
                                    "name": "CPU (%)",
                                    "query": "sum(rate(container_cpu_usage_seconds_total{name=~\".*caliper.*\"}[1m])) by (name) * 100",
                                    "step": 1,
                                    "label": "name",
                                    "statistic": "avg"
                                }
                            ]
                        }
                    }
                },
                {
                    "module": "docker",
                    "options": {
                        "interval": 1
                    }
                }
            ]
        }
        
        config_path = os.path.join(self.caliper_workspace, "benchmark-config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(benchmark_config, f, default_flow_style=False)
        
        self.benchmark_config = config_path
        logger.info(f"📄 Benchmark config generated: {config_path}")
    
    async def _generate_contract_bindings(self):
        """Tạo smart contract bindings cho Caliper"""
        
        # Create workloads directory
        workloads_dir = os.path.join(self.caliper_workspace, "workloads")
        os.makedirs(workloads_dir, exist_ok=True)
        
        # Register material workload
        register_workload = '''
'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');

class RegisterMaterialWorkload extends WorkloadModuleBase {
    constructor() {
        super();
    }

    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
        
        this.contractName = 'SpectroChain';
        this.roundIndex = roundIndex;
    }

    async submitTransaction() {
        const productId = `PRODUCT_${Math.random().toString(36).substr(2, 9)}`;
        const batchId = `BATCH_${Math.random().toString(36).substr(2, 9)}`;
        const dataHash = `0x${Buffer.from(Math.random().toString()).toString('hex')}`;
        
        const request = {
            contractId: this.contractName,
            contractFunction: 'registerMaterial',
            contractArguments: [productId, batchId, dataHash],
            readOnly: false
        };

        await this.sutAdapter.sendRequests(request);
    }
}

function createWorkloadModule() {
    return new RegisterMaterialWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;
'''
        
        with open(os.path.join(workloads_dir, "register-material.js"), 'w') as f:
            f.write(register_workload)
        
        # Verify material workload
        verify_workload = '''
'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');

class VerifyMaterialWorkload extends WorkloadModuleBase {
    constructor() {
        super();
        this.productIds = [];
    }

    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
        
        this.contractName = 'SpectroChain';
        this.roundIndex = roundIndex;
        
        // Pre-populate some product IDs for verification
        for (let i = 0; i < 100; i++) {
            this.productIds.push(`PRODUCT_${Math.random().toString(36).substr(2, 9)}`);
        }
    }

    async submitTransaction() {
        const productId = this.productIds[Math.floor(Math.random() * this.productIds.length)];
        
        const request = {
            contractId: this.contractName,
            contractFunction: 'verifyMaterial',
            contractArguments: [productId],
            readOnly: true
        };

        await this.sutAdapter.sendRequests(request);
    }
}

function createWorkloadModule() {
    return new VerifyMaterialWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;
'''
        
        with open(os.path.join(workloads_dir, "verify-material.js"), 'w') as f:
            f.write(verify_workload)
        
        # Mixed workload
        mixed_workload = '''
'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');

class MixedWorkload extends WorkloadModuleBase {
    constructor() {
        super();
        this.productIds = [];
    }

    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
        
        this.contractName = 'SpectroChain';
        this.roundIndex = roundIndex;
        
        // Pre-populate some product IDs
        for (let i = 0; i < 50; i++) {
            this.productIds.push(`PRODUCT_${Math.random().toString(36).substr(2, 9)}`);
        }
    }

    async submitTransaction() {
        // 70% registration, 30% verification
        if (Math.random() < 0.7) {
            // Register material
            const productId = `PRODUCT_${Math.random().toString(36).substr(2, 9)}`;
            const batchId = `BATCH_${Math.random().toString(36).substr(2, 9)}`;
            const dataHash = `0x${Buffer.from(Math.random().toString()).toString('hex')}`;
            
            this.productIds.push(productId);
            
            const request = {
                contractId: this.contractName,
                contractFunction: 'registerMaterial',
                contractArguments: [productId, batchId, dataHash],
                readOnly: false
            };

            await this.sutAdapter.sendRequests(request);
        } else {
            // Verify material
            if (this.productIds.length > 0) {
                const productId = this.productIds[Math.floor(Math.random() * this.productIds.length)];
                
                const request = {
                    contractId: this.contractName,
                    contractFunction: 'verifyMaterial',
                    contractArguments: [productId],
                    readOnly: true
                };

                await this.sutAdapter.sendRequests(request);
            }
        }
    }
}

function createWorkloadModule() {
    return new MixedWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;
'''
        
        with open(os.path.join(workloads_dir, "mixed-operations.js"), 'w') as f:
            f.write(mixed_workload)
        
        logger.info("📄 Contract workloads generated")
    
    async def run_caliper_benchmark(self) -> Dict[str, Any]:
        """Chạy Caliper benchmark"""
        logger.info("🚀 Running Caliper benchmark...")
        
        try:
            # Setup workspace if not exists
            if not os.path.exists(self.caliper_workspace):
                await self.setup_caliper_workspace()
            
            # Run Caliper command
            cmd = [
                "npx", "caliper", "launch", "manager",
                "--caliper-workspace", self.caliper_workspace,
                "--caliper-networkconfig", self.network_config,
                "--caliper-benchconfig", self.benchmark_config,
                "--caliper-flow-only-test"
            ]
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Run the benchmark
            result = subprocess.run(
                cmd,
                cwd=self.caliper_workspace,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Caliper benchmark completed successfully")
                return await self._parse_caliper_results()
            else:
                logger.error(f"❌ Caliper benchmark failed: {result.stderr}")
                return {
                    "error": "Caliper benchmark failed",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        
        except subprocess.TimeoutExpired:
            logger.error("⏰ Caliper benchmark timed out")
            return {"error": "Benchmark timed out"}
        
        except Exception as e:
            logger.error(f"💥 Error running Caliper: {e}")
            return {"error": str(e)}
    
    async def _parse_caliper_results(self) -> Dict[str, Any]:
        """Parse kết quả từ Caliper"""
        
        # Look for Caliper report files
        report_files = []
        for root, dirs, files in os.walk(self.caliper_workspace):
            for file in files:
                if file.endswith('.html') or file.endswith('.json'):
                    if 'report' in file.lower() or 'result' in file.lower():
                        report_files.append(os.path.join(root, file))
        
        results = {
            "caliper_completed": True,
            "report_files": report_files,
            "summary": {}
        }
        
        # Try to parse JSON results if available
        for file_path in report_files:
            if file_path.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        caliper_data = json.load(f)
                    
                    # Extract key metrics
                    if 'rounds' in caliper_data:
                        for round_data in caliper_data['rounds']:
                            round_name = round_data.get('label', 'Unknown')
                            performance = round_data.get('performance', {})
                            
                            results['summary'][round_name] = {
                                'tps': performance.get('tps', 0),
                                'latency_avg': performance.get('latency', {}).get('avg', 0),
                                'latency_max': performance.get('latency', {}).get('max', 0),
                                'success_rate': performance.get('success_rate', 0),
                                'throughput': performance.get('throughput', 0)
                            }
                    
                    results['raw_caliper_data'] = caliper_data
                    break
                    
                except Exception as e:
                    logger.error(f"Error parsing Caliper results: {e}")
        
        return results
    
    def generate_caliper_docker_compose(self):
        """Tạo Docker Compose file cho Ganache"""
        
        docker_compose = '''
version: '3.8'

services:
  ganache:
    image: trufflesuite/ganache-cli:latest
    ports:
      - "7545:8545"
    command: >
      --deterministic
      --accounts 10
      --host 0.0.0.0
      --port 8545
      --networkId 5777
      --gasLimit 10000000
      --gasPrice 20000000000
      --defaultBalanceEther 1000
    networks:
      - caliper-network

networks:
  caliper-network:
    driver: bridge
'''
        
        compose_path = os.path.join(self.caliper_workspace, "docker-compose-ganache.yaml")
        with open(compose_path, 'w') as f:
            f.write(docker_compose)
        
        logger.info(f"🐳 Docker Compose generated: {compose_path}")
    
    def install_caliper_dependencies(self):
        """Cài đặt Caliper dependencies"""
        logger.info("📦 Installing Caliper dependencies...")
        
        try:
            # Create package.json for Caliper workspace
            package_json = {
                "name": "spectrochain-caliper-workspace",
                "version": "1.0.0",
                "description": "Caliper workspace for SpectroChain benchmarking",
                "dependencies": {
                    "@hyperledger/caliper-cli": "0.5.0",
                    "@hyperledger/caliper-core": "0.5.0",
                    "@hyperledger/caliper-ethereum": "0.5.0"
                }
            }
            
            package_path = os.path.join(self.caliper_workspace, "package.json")
            with open(package_path, 'w') as f:
                json.dump(package_json, f, indent=2)
            
            # Install dependencies
            subprocess.run(
                ["npm", "install"],
                cwd=self.caliper_workspace,
                check=True
            )
            
            logger.info("✅ Caliper dependencies installed")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install Caliper dependencies: {e}")
            raise
    
    async def get_caliper_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Tạo khuyến nghị dựa trên kết quả Caliper"""
        recommendations = []
        
        if 'summary' in results:
            for round_name, metrics in results['summary'].items():
                tps = metrics.get('tps', 0)
                latency = metrics.get('latency_avg', 0)
                success_rate = metrics.get('success_rate', 1)
                
                if tps < 10:
                    recommendations.append(f"Low TPS in {round_name}: Consider optimizing smart contract gas usage")
                
                if latency > 1000:  # 1 second
                    recommendations.append(f"High latency in {round_name}: Check network configuration and node performance")
                
                if success_rate < 0.95:
                    recommendations.append(f"Low success rate in {round_name}: Investigate transaction failures")
        
        if not recommendations:
            recommendations.append("Caliper benchmark results look good - system performing within expected parameters")
        
        return recommendations
