# SpectroChain-Dental Benchmark Suite

Hệ thống đánh giá toàn diện cho hiệu năng, bảo mật và độ chính xác của SpectroChain-Dental.

## 🎯 Mục tiêu

Benchmark suite này đánh giá hệ thống theo các tiêu chí:

### 📊 Hiệu Năng (Performance)

- **Thông lượng (TPS)**: Giao dịch/giây
- **Độ trễ (Latency)**: Thời gian phản hồi
- **Sử dụng tài nguyên**: CPU, RAM, Disk
- **Khả năng mở rộng**: Hiệu suất với tải tăng dần

### 🔒 Bảo Mật (Security)

- **Phân tích STRIDE**: Đánh giá 6 loại nguy cơ bảo mật
- **Mô phỏng tấn công**: Selfish Mining, Double Spending, Eclipse Attack
- **Phân tích ngưỡng bảo mật**: Khả năng chống tập trung hóa
- **Đánh giá consensus**: Độ mạnh của thuật toán đồng thuận

### 🎯 Độ Chính Xác (Accuracy)

- **Xác minh vật liệu**: Precision, Recall, F1-score
- **Machine Learning Models**: SVM, Random Forest, Neural Network
- **Cross-validation**: Đánh giá độ tin cậy
- **Hash Quality Index (HQI)**: Chất lượng cryptographic hash

## 🚀 Cài đặt

### 1. Cài đặt dependencies

```bash
cd benchmark
pip install -r requirements.txt
```

### 2. Cấu hình

Chỉnh sửa `benchmark_config.json` theo nhu cầu:

```json
{
  "performance": {
    "concurrent_users": [1, 5, 10, 20, 50],
    "test_duration": 300
  },
  "security": {
    "attack_types": ["selfish_mining", "double_spending", "eclipse"],
    "threshold_tests": [0.1, 0.25, 0.33, 0.5]
  },
  "accuracy": {
    "models": ["svm", "random_forest", "neural_network"],
    "cross_validation_folds": 5
  }
}
```

## 🔄 Sử dụng

### Chạy benchmark đầy đủ

```bash
python main.py --mode full
```

### Chạy từng module riêng lẻ

```bash
# Chỉ test hiệu năng
python main.py --mode performance

# Chỉ test bảo mật
python main.py --mode security

# Chỉ test độ chính xác
python main.py --mode accuracy
```

### Với cấu hình tùy chỉnh

```bash
python main.py --mode full --config my_config.json --output-dir my_results
```

## 📊 Kết quả

### 1. Báo cáo HTML

Được tạo tự động trong thư mục `results/`:

- Tổng quan hiệu năng
- Phân tích bảo mật chi tiết
- Metrics độ chính xác
- Visualizations interactive

### 2. Dashboard interactiv

Sử dụng Plotly để tạo dashboard realtime:

- Biểu đồ TPS theo thời gian
- STRIDE security analysis
- Model performance comparison

### 3. Raw Data

Dữ liệu JSON chi tiết cho phân tích sâu hơn.

## 🏗️ Kiến trúc

```
benchmark/
├── main.py                 # Entry point chính
├── benchmark_config.json   # Cấu hình benchmark
├── requirements.txt        # Dependencies
├── performance/
│   └── performance_tester.py  # Test hiệu năng
├── security/
│   └── security_analyzer.py   # Phân tích bảo mật
├── accuracy/
│   └── accuracy_evaluator.py  # Đánh giá độ chính xác
├── utils/
│   ├── metrics_collector.py   # Thu thập metrics
│   └── report_generator.py    # Tạo báo cáo
└── results/               # Kết quả benchmark
```

## 📈 Metrics Chi Tiết

### Performance Metrics

- **TPS (Transactions Per Second)**: Đo khả năng xử lý giao dịch
- **Latency**: P50, P95, P99 percentiles
- **Resource Utilization**: CPU, Memory, Disk I/O
- **Scalability Patterns**: Linear, Degraded, Poor

### Security Metrics

- **STRIDE Score**: 0-1 cho mỗi threat category
- **Attack Success Rate**: Xác suất thành công các cuộc tấn công
- **Consensus Strength**: Strong, Medium, Weak
- **Decentralization Index**: Gini coefficient

### Accuracy Metrics

- **Verification Accuracy**: True Positive Rate cho xác minh vật liệu
- **ML Model Performance**: Accuracy, Precision, Recall, F1
- **Cross-validation Score**: Mean ± Standard Deviation
- **Hash Quality Index**: Entropy và collision resistance

## 🎚️ Thresholds và Grading

### Performance Grading

- **A Grade**: TPS > 50, Latency < 100ms, CPU < 70%
- **B Grade**: TPS > 20, Latency < 500ms, CPU < 80%
- **C Grade**: TPS > 10, Latency < 1000ms, CPU < 90%

### Security Grading

- **A Grade**: Overall Security Score > 0.9
- **B Grade**: Overall Security Score > 0.7
- **C Grade**: Overall Security Score > 0.5

### Accuracy Grading

- **A+ Grade**: F1 Score > 0.95
- **A Grade**: F1 Score > 0.90
- **B Grade**: F1 Score > 0.80

## 🔧 Tùy chỉnh

### Thêm test cases mới

1. Tạo module trong thư mục tương ứng
2. Implement interface chuẩn
3. Cập nhật config
4. Chạy benchmark

### Thêm metrics mới

1. Extend MetricsCollector
2. Update ReportGenerator
3. Modify HTML templates

## 🚨 Lưu ý

### Prerequisites

- Python 3.8+
- SpectroChain-Dental đang chạy
- Ganache blockchain network
- Đủ tài nguyên hệ thống cho load testing

### Performance Impact

- Benchmark có thể gây tải cao lên hệ thống
- Chạy trong môi trường test riêng biệt
- Monitor resource usage during testing

### Security Testing

- Chỉ chạy attack simulations trong môi trường an toàn
- Không test trên production
- Backup dữ liệu trước khi test

## 📚 Tài liệu tham khảo

### Methodologies

- STRIDE Threat Modeling (Microsoft)
- NIST Cybersecurity Framework
- OWASP Security Testing Guide

### Standards

- ISO 27001 (Information Security)
- FDA UDI Requirements
- Blockchain Security Best Practices

## 🤝 Contribution

1. Fork repository
2. Tạo feature branch
3. Implement và test
4. Submit pull request

## 📞 Hỗ trợ

- GitHub Issues cho bug reports
- Documentation wiki
- Email support: [support@spectrochain.com]

---

**Lưu ý**: Benchmark suite này được thiết kế để đánh giá toàn diện hệ thống SpectroChain-Dental. Kết quả cần được phân tích trong context cụ thể của từng deployment environment.
