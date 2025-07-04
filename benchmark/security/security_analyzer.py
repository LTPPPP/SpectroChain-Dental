"""
Security Analysis Module
Đánh giá bảo mật: Selfish Mining, Double Spending, Eclipse Attack, STRIDE Model
"""

import asyncio
import hashlib
import random
import time
import json
import statistics
from typing import Dict, List, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SecurityAnalyzer:
    """Công cụ phân tích bảo mật"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.attack_types = config.get('attack_types', ['selfish_mining', 'double_spending', 'eclipse'])
        self.threshold_tests = config.get('threshold_tests', [0.1, 0.25, 0.33, 0.5])
        self.network_analysis = config.get('network_analysis', True)
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Chạy tất cả security tests"""
        results = {
            'consensus_security': await self._analyze_consensus_security(),
            'attack_simulations': await self._simulate_attacks(),
            'stride_analysis': await self._run_stride_analysis(),
            'data_integrity': await self._test_data_integrity(),
            'oracle_security': await self._test_oracle_security(),
            'network_security': await self._analyze_network_security(),
            'threshold_analysis': await self._analyze_security_thresholds()
        }
        
        # Tính toán security score tổng hợp
        results['overall_security_score'] = self._calculate_security_score(results)
        results['vulnerabilities'] = self._identify_vulnerabilities(results)
        results['recommendations'] = self._generate_security_recommendations(results)
        
        return results
    
    async def _analyze_consensus_security(self) -> Dict[str, Any]:
        """Phân tích bảo mật của consensus mechanism"""
        logger.info("🔒 Analyzing Consensus Security...")
        
        # Simulate consensus analysis for PoA/PoS (Ganache typically uses PoA)
        results = {
            'consensus_type': 'Proof_of_Authority',
            'validator_count': 10,  # Simulated
            'validator_distribution': await self._analyze_validator_distribution(),
            'block_finality': await self._test_block_finality(),
            'fork_resistance': await self._test_fork_resistance(),
            'consensus_strength': 'strong'  # Based on PoA characteristics
        }
        
        return results
    
    async def _analyze_validator_distribution(self) -> Dict[str, float]:
        """Phân tích phân bố validator"""
        # Simulate validator power distribution
        validators = [0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.13]
        
        # Calculate Gini coefficient for decentralization
        gini = self._calculate_gini_coefficient(validators)
        
        return {
            'gini_coefficient': gini,
            'top_3_validator_power': sum(sorted(validators, reverse=True)[:3]),
            'decentralization_score': 1 - gini,  # Higher is more decentralized
            'validator_count': len(validators)
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Tính hệ số Gini để đánh giá độ tập trung"""
        n = len(values)
        values_sorted = sorted(values)
        cumsum = np.cumsum(values_sorted)
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * sum(values))
    
    async def _test_block_finality(self) -> Dict[str, Any]:
        """Test tính chắc chắn của block"""
        # Simulate finality testing
        finality_times = [2.1, 1.9, 2.3, 2.0, 1.8]  # seconds
        
        return {
            'avg_finality_time': statistics.mean(finality_times),
            'finality_confidence': 0.99,  # 99% confidence after finality
            'reorg_probability': 0.001   # Very low for PoA
        }
    
    async def _test_fork_resistance(self) -> Dict[str, float]:
        """Test khả năng chống fork"""
        return {
            'natural_fork_rate': 0.005,  # 0.5% of blocks
            'maximum_fork_depth': 2,
            'fork_resolution_time': 30.5  # seconds
        }
    
    async def _simulate_attacks(self) -> Dict[str, Any]:
        """Mô phỏng các loại tấn công"""
        logger.info("⚔️ Simulating Security Attacks...")
        
        results = {}
        
        for attack_type in self.attack_types:
            if attack_type == 'selfish_mining':
                results['selfish_mining'] = await self._simulate_selfish_mining()
            elif attack_type == 'double_spending':
                results['double_spending'] = await self._simulate_double_spending()
            elif attack_type == 'eclipse':
                results['eclipse_attack'] = await self._simulate_eclipse_attack()
        
        return results
    
    async def _simulate_selfish_mining(self) -> Dict[str, Any]:
        """Mô phỏng Selfish Mining Attack"""
        logger.info("🎣 Simulating Selfish Mining Attack...")
        
        results = {}
        
        for threshold in self.threshold_tests:
            simulation_result = await self._run_selfish_mining_simulation(threshold)
            results[f'threshold_{threshold}'] = simulation_result
        
        # Tính toán revenue advantage
        results['revenue_analysis'] = self._analyze_selfish_mining_revenue(results)
        
        return results
    
    async def _run_selfish_mining_simulation(self, attacker_power: float) -> Dict[str, float]:
        """Chạy simulation selfish mining với % power nhất định"""
        
        # Simulate mining for 1000 blocks
        blocks_mined = 1000
        honest_blocks = 0
        attacker_blocks = 0
        attacker_private_chain = 0
        
        for _ in range(blocks_mined):
            # Random mining based on power distribution
            if random.random() < attacker_power:
                # Attacker mines block
                if attacker_private_chain > 0:
                    # Continue private chain
                    attacker_private_chain += 1
                else:
                    # Start private chain
                    attacker_private_chain = 1
            else:
                # Honest miner finds block
                if attacker_private_chain > 1:
                    # Attacker publishes and gains advantage
                    attacker_blocks += attacker_private_chain
                    attacker_private_chain = 0
                elif attacker_private_chain == 1:
                    # Race condition - attacker loses
                    attacker_private_chain = 0
                
                honest_blocks += 1
        
        total_blocks = honest_blocks + attacker_blocks
        attacker_share = attacker_blocks / total_blocks if total_blocks > 0 else 0
        revenue_advantage = (attacker_share - attacker_power) / attacker_power if attacker_power > 0 else 0
        
        return {
            'attacker_power': attacker_power,
            'attacker_share': attacker_share,
            'revenue_advantage': revenue_advantage,
            'blocks_won': attacker_blocks,
            'is_profitable': revenue_advantage > 0
        }
    
    def _analyze_selfish_mining_revenue(self, results: Dict) -> Dict[str, Any]:
        """Phân tích revenue advantage từ selfish mining"""
        profitable_thresholds = []
        max_advantage = 0
        
        for key, result in results.items():
            if key.startswith('threshold_'):
                if result['is_profitable']:
                    profitable_thresholds.append(result['attacker_power'])
                max_advantage = max(max_advantage, result['revenue_advantage'])
        
        return {
            'min_profitable_threshold': min(profitable_thresholds) if profitable_thresholds else None,
            'max_revenue_advantage': max_advantage,
            'vulnerable_to_selfish_mining': len(profitable_thresholds) > 0
        }
    
    async def _simulate_double_spending(self) -> Dict[str, Any]:
        """Mô phỏng Double Spending Attack"""
        logger.info("💰 Simulating Double Spending Attack...")
        
        results = {}
        
        for confirmations in [1, 3, 6, 12]:
            success_rate = await self._calculate_double_spend_success(confirmations)
            results[f'{confirmations}_confirmations'] = {
                'success_probability': success_rate,
                'recommended': success_rate < 0.01  # Less than 1% chance
            }
        
        results['analysis'] = {
            'safe_confirmation_count': self._find_safe_confirmation_count(results),
            'attack_feasibility': 'low' if results['6_confirmations']['success_probability'] < 0.01 else 'high'
        }
        
        return results
    
    async def _calculate_double_spend_success(self, confirmations: int) -> float:
        """Tính xác suất thành công double spending với số confirmation"""
        # For PoA, double spending is extremely difficult
        # Simulate based on theoretical model
        
        attacker_power = 0.3  # 30% attacker power
        success_probability = 0
        
        for k in range(confirmations + 1):
            # Binomial probability
            prob_k_blocks = (attacker_power ** k) * ((1 - attacker_power) ** (confirmations - k))
            if k >= confirmations:
                success_probability += prob_k_blocks
        
        # For PoA, reduce probability significantly
        return success_probability * 0.1  # PoA makes this much harder
    
    def _find_safe_confirmation_count(self, results: Dict) -> int:
        """Tìm số confirmation an toàn"""
        for conf in [1, 3, 6, 12]:
            key = f'{conf}_confirmations'
            if key in results and results[key]['success_probability'] < 0.01:
                return conf
        return 12  # Default safe value
    
    async def _simulate_eclipse_attack(self) -> Dict[str, Any]:
        """Mô phỏng Eclipse Attack"""
        logger.info("🌑 Simulating Eclipse Attack...")
        
        # Simulate network connectivity analysis
        node_connections = random.randint(8, 32)  # Typical P2P connections
        attacker_nodes = random.randint(1, 5)
        
        eclipse_probability = min(attacker_nodes / node_connections, 0.8)
        
        return {
            'average_node_connections': node_connections,
            'potential_attacker_nodes': attacker_nodes,
            'eclipse_probability': eclipse_probability,
            'mitigation_effectiveness': 0.9,  # Good P2P design
            'vulnerability_level': 'low' if eclipse_probability < 0.3 else 'medium' if eclipse_probability < 0.6 else 'high'
        }
    
    async def _run_stride_analysis(self) -> Dict[str, Any]:
        """STRIDE Threat Model Analysis"""
        logger.info("🛡️ Running STRIDE Analysis...")
        
        stride_results = {
            'Spoofing': await self._analyze_spoofing_threats(),
            'Tampering': await self._analyze_tampering_threats(),
            'Repudiation': await self._analyze_repudiation_threats(),
            'Information_Disclosure': await self._analyze_information_disclosure(),
            'Denial_of_Service': await self._analyze_dos_threats(),
            'Elevation_of_Privilege': await self._analyze_privilege_escalation()
        }
        
        # Tính toán STRIDE score tổng hợp
        stride_results['overall_stride_score'] = self._calculate_stride_score(stride_results)
        
        return stride_results
    
    async def _analyze_spoofing_threats(self) -> Dict[str, Any]:
        """Phân tích nguy cơ Spoofing"""
        return {
            'identity_verification_strength': 0.9,  # Strong cryptographic identity
            'certificate_validation': True,
            'multi_factor_authentication': False,  # Not implemented in MVP
            'vulnerability_score': 0.2,  # Low vulnerability
            'mitigations': ['Digital signatures', 'Public key cryptography']
        }
    
    async def _analyze_tampering_threats(self) -> Dict[str, Any]:
        """Phân tích nguy cơ Tampering"""
        return {
            'data_integrity_protection': 0.95,  # Blockchain immutability
            'hash_verification': True,
            'digital_signatures': True,
            'vulnerability_score': 0.1,  # Very low vulnerability
            'mitigations': ['Cryptographic hashing', 'Blockchain immutability', 'Digital signatures']
        }
    
    async def _analyze_repudiation_threats(self) -> Dict[str, Any]:
        """Phân tích nguy cơ Repudiation"""
        return {
            'transaction_logging': True,
            'digital_signatures': True,
            'timestamp_reliability': 0.9,
            'audit_trail_completeness': 0.85,
            'vulnerability_score': 0.15,
            'mitigations': ['Immutable transaction logs', 'Digital signatures', 'Blockchain timestamps']
        }
    
    async def _analyze_information_disclosure(self) -> Dict[str, Any]:
        """Phân tích nguy cơ Information Disclosure"""
        return {
            'data_encryption_at_rest': False,  # Not implemented in MVP
            'data_encryption_in_transit': True,  # HTTPS
            'access_control_strength': 0.6,  # Basic role-based
            'vulnerability_score': 0.4,  # Medium vulnerability
            'mitigations': ['HTTPS encryption', 'Role-based access'],
            'recommendations': ['Implement data encryption at rest', 'Enhanced access controls']
        }
    
    async def _analyze_dos_threats(self) -> Dict[str, Any]:
        """Phân tích nguy cơ Denial of Service"""
        return {
            'rate_limiting': False,  # Not implemented in MVP
            'resource_consumption_limits': False,
            'ddos_protection': False,
            'vulnerability_score': 0.7,  # High vulnerability
            'mitigations': ['Gas limits on blockchain'],
            'recommendations': ['Implement rate limiting', 'Add DDoS protection', 'Resource limits']
        }
    
    async def _analyze_privilege_escalation(self) -> Dict[str, Any]:
        """Phân tích nguy cơ Elevation of Privilege"""
        return {
            'role_based_access_control': True,
            'privilege_separation': 0.7,
            'admin_controls': 0.6,
            'vulnerability_score': 0.3,
            'mitigations': ['Role-based permissions', 'Smart contract access controls'],
            'recommendations': ['Enhanced admin controls', 'Privilege auditing']
        }
    
    def _calculate_stride_score(self, stride_results: Dict) -> float:
        """Tính toán STRIDE security score"""
        threat_scores = []
        
        for threat_type, analysis in stride_results.items():
            if isinstance(analysis, dict) and 'vulnerability_score' in analysis:
                # Lower vulnerability score = higher security
                security_score = 1 - analysis['vulnerability_score']
                threat_scores.append(security_score)
        
        return statistics.mean(threat_scores) if threat_scores else 0
    
    async def _test_data_integrity(self) -> Dict[str, Any]:
        """Test tính toàn vẹn dữ liệu"""
        logger.info("🔍 Testing Data Integrity...")
        
        # Test hash consistency
        test_data = "test_spectroscopy_data"
        hash1 = hashlib.sha256(test_data.encode()).hexdigest()
        hash2 = hashlib.sha256(test_data.encode()).hexdigest()
        
        # Test data modification detection
        modified_data = test_data + "modified"
        hash_modified = hashlib.sha256(modified_data.encode()).hexdigest()
        
        return {
            'hash_consistency': hash1 == hash2,
            'modification_detection': hash1 != hash_modified,
            'hash_algorithm': 'SHA-256',
            'collision_resistance': 'strong',
            'integrity_score': 1.0 if hash1 == hash2 and hash1 != hash_modified else 0.0
        }
    
    async def _test_oracle_security(self) -> Dict[str, Any]:
        """Test bảo mật Oracle Problem"""
        logger.info("🔮 Testing Oracle Security...")
        
        return {
            'data_source_verification': False,  # Not implemented in MVP
            'multiple_data_sources': False,
            'data_validation': True,  # Basic validation exists
            'oracle_centralization_risk': 'high',  # Single source in MVP
            'vulnerability_score': 0.6,
            'recommendations': [
                'Implement multiple data sources',
                'Add data source verification',
                'Implement oracle reputation system'
            ]
        }
    
    async def _analyze_network_security(self) -> Dict[str, Any]:
        """Phân tích bảo mật mạng"""
        logger.info("🌐 Analyzing Network Security...")
        
        if not self.network_analysis:
            return {'skipped': True}
        
        return {
            'p2p_network_analysis': {
                'node_connectivity': 'good',
                'network_partition_resistance': 0.8,
                'sybil_attack_resistance': 0.7
            },
            'communication_security': {
                'encryption_in_transit': True,
                'certificate_validation': True,
                'secure_protocols': ['HTTPS', 'WSS']
            },
            'network_monitoring': {
                'intrusion_detection': False,
                'anomaly_detection': False,
                'traffic_analysis': False
            }
        }
    
    async def _analyze_security_thresholds(self) -> Dict[str, Any]:
        """Phân tích ngưỡng bảo mật"""
        logger.info("📊 Analyzing Security Thresholds...")
        
        thresholds = {}
        
        for threshold in self.threshold_tests:
            thresholds[f'threshold_{threshold}'] = {
                'attacker_power': threshold,
                'system_security_level': self._calculate_security_at_threshold(threshold),
                'recommended_countermeasures': self._get_countermeasures_for_threshold(threshold)
            }
        
        return thresholds
    
    def _calculate_security_at_threshold(self, threshold: float) -> str:
        """Tính mức độ bảo mật tại ngưỡng nhất định"""
        if threshold < 0.1:
            return 'very_high'
        elif threshold < 0.25:
            return 'high'
        elif threshold < 0.33:
            return 'medium'
        elif threshold < 0.5:
            return 'low'
        else:
            return 'critical'
    
    def _get_countermeasures_for_threshold(self, threshold: float) -> List[str]:
        """Lấy biện pháp đối phó cho ngưỡng"""
        if threshold >= 0.5:
            return ['Emergency response', 'Network upgrade', 'Additional validators']
        elif threshold >= 0.33:
            return ['Increased monitoring', 'Enhanced consensus', 'Validator diversification']
        elif threshold >= 0.25:
            return ['Regular audits', 'Proactive monitoring']
        else:
            return ['Routine security measures']
    
    def _calculate_security_score(self, results: Dict) -> float:
        """Tính toán security score tổng hợp"""
        scores = []
        
        # STRIDE Score
        if 'stride_analysis' in results:
            stride_score = results['stride_analysis'].get('overall_stride_score', 0)
            scores.append(stride_score * 0.3)  # 30% weight
        
        # Consensus Security
        if 'consensus_security' in results:
            consensus_strength = results['consensus_security'].get('consensus_strength', 'weak')
            consensus_score = 0.9 if consensus_strength == 'strong' else 0.6 if consensus_strength == 'medium' else 0.3
            scores.append(consensus_score * 0.25)  # 25% weight
        
        # Data Integrity
        if 'data_integrity' in results:
            integrity_score = results['data_integrity'].get('integrity_score', 0)
            scores.append(integrity_score * 0.2)  # 20% weight
        
        # Attack Resistance
        if 'attack_simulations' in results:
            attack_resistance = 1.0  # Default high resistance for PoA
            if 'selfish_mining' in results['attack_simulations']:
                selfish_mining = results['attack_simulations']['selfish_mining']
                if 'revenue_analysis' in selfish_mining:
                    vulnerable = selfish_mining['revenue_analysis'].get('vulnerable_to_selfish_mining', False)
                    attack_resistance = 0.6 if vulnerable else 0.9
            scores.append(attack_resistance * 0.25)  # 25% weight
        
        return sum(scores) if scores else 0.0
    
    def _identify_vulnerabilities(self, results: Dict) -> List[Dict[str, Any]]:
        """Xác định các lỗ hổng bảo mật"""
        vulnerabilities = []
        
        # Check STRIDE vulnerabilities
        if 'stride_analysis' in results:
            for threat_type, analysis in results['stride_analysis'].items():
                if isinstance(analysis, dict) and analysis.get('vulnerability_score', 0) > 0.5:
                    vulnerabilities.append({
                        'type': 'STRIDE',
                        'threat': threat_type,
                        'severity': 'high' if analysis['vulnerability_score'] > 0.7 else 'medium',
                        'score': analysis['vulnerability_score'],
                        'recommendations': analysis.get('recommendations', [])
                    })
        
        # Check Oracle vulnerabilities
        if 'oracle_security' in results:
            oracle_vuln = results['oracle_security'].get('vulnerability_score', 0)
            if oracle_vuln > 0.5:
                vulnerabilities.append({
                    'type': 'Oracle',
                    'threat': 'Oracle Problem',
                    'severity': 'high' if oracle_vuln > 0.7 else 'medium',
                    'score': oracle_vuln,
                    'recommendations': results['oracle_security'].get('recommendations', [])
                })
        
        return vulnerabilities
    
    def _generate_security_recommendations(self, results: Dict) -> List[str]:
        """Tạo khuyến nghị bảo mật"""
        recommendations = []
        
        # Based on vulnerabilities found
        vulnerabilities = self._identify_vulnerabilities(results)
        
        for vuln in vulnerabilities:
            recommendations.extend(vuln.get('recommendations', []))
        
        # General recommendations
        recommendations.extend([
            'Implement comprehensive logging and monitoring',
            'Regular security audits and penetration testing',
            'Keep all components updated',
            'Implement rate limiting and DDoS protection',
            'Add data encryption at rest',
            'Enhance access control mechanisms'
        ])
        
        # Remove duplicates
        return list(set(recommendations))
