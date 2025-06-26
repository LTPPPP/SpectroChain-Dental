# 🎯 Evaluation Directory

Thư mục này chứa tất cả các công cụ đánh giá và metrics cho SpectroChain-Dental.

## 📁 Cấu trúc thư mục

### `/metrics/`
- **Chức năng**: Scripts tính toán metrics chính
- **Files**:
  - `performance_metrics.py` - Đánh giá hiệu suất tổng thể
  - `blockchain_metrics.py` - Metrics blockchain (sẽ có)
  - `security_metrics.py` - Metrics bảo mật (sẽ có)
  - `accuracy_metrics.py` - Metrics độ chính xác (sẽ có)

### `/algorithms/`
- **Chức năng**: Thuật toán đánh giá và scoring
- **Files sẽ có**:
  - `scoring_algorithms.py` - Thuật toán tính điểm
  - `weight_calculation.py` - Tính trọng số metrics
  - `normalization.py` - Chuẩn hóa dữ liệu
  - `comparative_scoring.py` - Tính điểm so sánh

## 🎯 Các loại Evaluation

### 1. Performance Evaluation
- **Blockchain Performance**: TPS, Latency, Resource Usage
- **Function Breakdown**: registerMaterial, transferOwnership, verifyMaterial
- **System Resource**: CPU, Memory monitoring

### 2. Security Evaluation
- **STRIDE Analysis**: 6 threat categories
- **Penetration Testing**: Real attack simulation
- **Vulnerability Assessment**: Security scoring

### 3. Accuracy Evaluation
- **Physical Verification**: Spectral analysis accuracy
- **Hit Quality Index**: Material matching precision
- **ML Metrics**: Precision, Recall, F1-Score, AUC

### 4. Comparative Evaluation
- **System Comparison**: Centralized vs Blockchain vs SpectroChain
- **Cross-validation**: Multiple verification methods
- **Benchmark Standards**: Industry comparison

## 🔢 Scoring Methodology

### Weight Distribution
```python
weights = {
    "throughput_tps": 0.15,           # 15% - Performance
    "latency_ms": 0.10,               # 10% - Responsiveness  
    "data_tamper_resistance": 0.20,   # 20% - Security
    "decentralized_trust": 0.20,      # 20% - Decentralization
    "physical_verification": 0.25,    # 25% - Unique Feature
    "oracle_resilience": 0.10         # 10% - Problem Solving
}
```

### Normalization
- **Throughput**: Scale to 0-100 (max reference: 200 TPS)
- **Latency**: Inverse scale (lower is better)
- **Security/Trust**: Direct percentage (0-100)
- **Physical Verification**: Accuracy percentage
- **Oracle Resilience**: Problem solving capability

## 🚀 Cách chạy Evaluation

### Full Evaluation
```bash
python evaluation/metrics/performance_metrics.py
```

### Specific Metrics
```bash
# Chỉ blockchain performance
python -c "from evaluation.metrics.performance_metrics import PerformanceEvaluator; PerformanceEvaluator().blockchain_performance_metrics()"

# Chỉ security analysis
python -c "from evaluation.metrics.performance_metrics import PerformanceEvaluator; PerformanceEvaluator().security_analysis()"
```

## 📊 Output Format

### JSON Results
```json
{
  "blockchain_performance": {...},
  "verification_accuracy": {...},
  "security_analysis": {...},
  "comparative_analysis": {...},
  "evaluation_summary": {...}
}
```

### Score Interpretation
- **90-100**: Excellent performance
- **80-89**: Good performance  
- **70-79**: Acceptable performance
- **60-69**: Needs improvement
- **<60**: Poor performance 