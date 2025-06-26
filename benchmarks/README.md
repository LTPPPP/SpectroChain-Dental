# 🏁 Benchmarks Directory

Thư mục này chứa tất cả các công cụ và scripts để benchmark hiệu suất hệ thống.

## 📁 Cấu trúc thư mục

### `/real_time/`
- **Chức năng**: Benchmark real-time cho tất cả hệ thống
- **Files**:
  - `benchmark_systems.py` - Core benchmark engine
  - `system_comparison.py` - So sánh cross-system (sẽ có)
  - `performance_profiler.py` - Profiling chi tiết (sẽ có)

### `/comparative/`
- **Chức năng**: So sánh với hệ thống khác
- **Files sẽ có**:
  - `centralized_benchmark.py` - Test hệ thống tập trung
  - `blockchain_benchmark.py` - Test blockchain thuần
  - `hybrid_benchmark.py` - Test hệ thống hybrid

### `/figures/` (Existing)
- **Chức năng**: Biểu đồ benchmark cũ
- **Files**: Các PNG files từ benchmark trước

### `/results/` (Existing) 
- **Chức năng**: Kết quả benchmark cũ
- **Files**: Dữ liệu benchmark lịch sử

## 🚀 Cách chạy Benchmark

### Quick Start
```bash
# Chạy benchmark tất cả hệ thống
python benchmarks/real_time/benchmark_systems.py

# Chạy performance metrics
python evaluation/metrics/performance_metrics.py

# Tạo biểu đồ
python results/charts/visualization_charts.py
```

### Advanced Testing
```bash
# Benchmark riêng lẻ
python -c "from benchmarks.real_time.benchmark_systems import RealTimeBenchmark; RealTimeBenchmark().benchmark_system('centralized')"

# Profile chi tiết
python benchmarks/real_time/performance_profiler.py --system=spectrochain_dental
```

## 📊 Loại Benchmark

1. **Throughput Testing**: TPS cho từng function
2. **Latency Measurement**: Response time thực tế
3. **Resource Monitoring**: CPU, Memory usage
4. **Security Penetration**: Tamper resistance testing
5. **Scalability Testing**: Load under stress
6. **Comparative Analysis**: Cross-system comparison

## 🎯 Benchmark Metrics

- **Performance**: TPS, Latency, Resource Usage
- **Security**: STRIDE compliance, Penetration results
- **Accuracy**: Physical verification precision
- **Reliability**: Error rates, Success rates
- **Scalability**: Load handling capacity 