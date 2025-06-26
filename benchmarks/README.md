# 🏁 Benchmarks Directory

This directory contains all tools and scripts for system performance benchmarking.

## 📁 Directory Structure

### `/real_time/`
- **Function**: Real-time benchmarking for all systems
- **Files**:
  - `benchmark_systems.py` - Core benchmark engine
  - `system_comparison.py` - Cross-system comparison (future)
  - `performance_profiler.py` - Detailed profiling (future)

### `/comparative/`
- **Function**: Comparison with other systems
- **Future Files**:
  - `centralized_benchmark.py` - Centralized system testing
  - `blockchain_benchmark.py` - Pure blockchain testing
  - `hybrid_benchmark.py` - Hybrid system testing

### `/figures/` (Existing)
- **Function**: Legacy benchmark charts
- **Files**: PNG files from previous benchmarks

### `/results/` (Existing) 
- **Function**: Legacy benchmark results
- **Files**: Historical benchmark data

## 🚀 Running Benchmarks

### Quick Start
```bash
# Run benchmark for all systems
python benchmarks/real_time/benchmark_systems.py

# Run performance metrics
python evaluation/metrics/performance_metrics.py

# Generate charts
python results/charts/visualization_charts.py
```

### Advanced Testing
```bash
# Individual system benchmark
python -c "from benchmarks.real_time.benchmark_systems import RealTimeBenchmark; RealTimeBenchmark().benchmark_system('centralized')"

# Detailed profiling
python benchmarks/real_time/performance_profiler.py --system=spectrochain_dental
```

### Command Line Options
```bash
# Full evaluation with verbose output
python run_evaluation.py --verbose

# Benchmark only (no charts)
python run_evaluation.py --benchmark-only

# Show latest results
python run_evaluation.py --show-results
```

## 📊 Benchmark Types

### 1. Performance Benchmarking
- **Throughput Testing**: TPS for each function
- **Latency Measurement**: Real response times
- **Resource Monitoring**: CPU and memory usage
- **Scalability Testing**: Load under stress

### 2. Security Benchmarking
- **Penetration Testing**: Attack resistance simulation
- **Tamper Resistance**: Data integrity verification
- **Access Control**: Permission validation
- **DoS Resistance**: Concurrent load testing

### 3. Accuracy Benchmarking
- **Physical Verification**: Spectral analysis precision
- **Material Recognition**: Authenticity detection
- **Noise Tolerance**: Performance under interference
- **False Positive/Negative**: Error rate analysis

### 4. Comparative Benchmarking
- **Cross-system Analysis**: Multiple platform comparison
- **Feature Parity**: Capability assessment
- **Performance Trade-offs**: Efficiency vs security
- **Scalability Comparison**: Growth potential analysis

## 🎯 Benchmark Metrics

### Performance Metrics
- **Throughput (TPS)**: Transactions per second
- **Latency (ms)**: Average response time
- **CPU Usage (%)**: Processor utilization
- **Memory Usage (MB)**: RAM consumption
- **Disk I/O**: Storage operations per second

### Security Metrics
- **STRIDE Scores**: Six threat category ratings
- **Penetration Success Rate**: Attack resistance percentage
- **Tamper Detection**: Data modification detection rate
- **Access Violations**: Unauthorized access attempts blocked

### Accuracy Metrics
- **Hit Quality Index (HQI)**: Material matching precision
- **Precision/Recall**: ML performance indicators
- **F1-Score**: Balanced accuracy measure
- **AUC**: Area under ROC curve

## 🔬 Benchmark Architecture

### System Under Test
```
┌─────────────────────────────────────┐
│        SpectroChain-Dental          │
├─────────────────────────────────────┤
│  Blockchain Layer    Spectral Layer │
│  ┌─────────────┐    ┌─────────────┐ │
│  │   Mining    │    │  Analysis   │ │
│  │ Consensus   │    │ Verification│ │
│  │Transaction  │    │  ML Engine  │ │
│  └─────────────┘    └─────────────┘ │
└─────────────────────────────────────┘
```

### Benchmark Components
1. **LoadGenerator**: Simulates realistic workloads
2. **MetricsCollector**: Gathers performance data
3. **SecurityTester**: Executes penetration tests
4. **AccuracyValidator**: Verifies output quality
5. **ResultsAggregator**: Compiles comprehensive reports

### Test Scenarios
- **Baseline Testing**: Single-user, optimal conditions
- **Load Testing**: Multiple concurrent users
- **Stress Testing**: Beyond normal capacity
- **Endurance Testing**: Extended operation periods
- **Peak Testing**: Maximum sustainable load

## 📈 Benchmark Results

### Current Performance (Latest Run)
| System | Overall Score | Throughput | Latency | Security |
|--------|---------------|------------|---------|----------|
| **SpectroChain-Dental** | **95.94/100** | 6,431.5 TPS | 12.3ms | 65/100 |
| Blockchain Only | 51.95/100 | 0.0 TPS | N/A | 100/100 |
| Centralized System | 31.61/100 | 22,503.99 TPS | 5.2ms | 20/100 |

### Performance Trends
- **Throughput**: Consistently above 6,000 TPS
- **Latency**: Stable under 15ms average
- **Resource Usage**: Optimal CPU/memory utilization
- **Accuracy**: Maintained 98%+ verification rate

### Security Analysis
- **Strengths**: Tamper resistance, data integrity
- **Improvements Needed**: DoS protection, spoofing prevention
- **Overall Rating**: Above average with optimization potential

## 🔧 Configuration Options

### Benchmark Parameters
```python
BENCHMARK_CONFIG = {
    "num_operations": 1000,
    "concurrent_users": 10,
    "test_duration": 300,  # seconds
    "warm_up_period": 30,
    "metrics_interval": 5,
    "security_tests": True,
    "accuracy_validation": True
}
```

### System Settings
```python
SYSTEM_CONFIG = {
    "blockchain_difficulty": 4,
    "mining_threads": 2,
    "verification_threshold": 0.95,
    "cache_size": 1000,
    "timeout_seconds": 30
}
```

## 🛠️ Development Guidelines

### Adding New Benchmarks
1. Extend `RealTimeBenchmark` class
2. Implement benchmark methods
3. Add metrics collection
4. Update result aggregation
5. Document test scenarios

### Modifying Existing Tests
1. Update benchmark parameters
2. Adjust performance thresholds
3. Enhance metrics collection
4. Validate result accuracy
5. Update documentation

### Performance Optimization
1. Identify bottlenecks through profiling
2. Optimize critical code paths
3. Implement caching strategies
4. Monitor resource utilization
5. Validate improvements through benchmarks

## 📝 Reporting Standards

### Automated Reports
- **JSON Format**: Machine-readable results
- **CSV Export**: Spreadsheet analysis
- **Markdown Summary**: Human-readable overview
- **Charts/Graphs**: Visual performance trends

### Manual Analysis
- **Performance Bottlenecks**: Identified issues
- **Optimization Opportunities**: Improvement areas
- **Comparative Analysis**: Competitive positioning
- **Recommendations**: Action items for enhancement 