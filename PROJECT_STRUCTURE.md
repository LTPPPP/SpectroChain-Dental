# 🏗️ SpectroChain-Dental Project Structure

## 📂 Tổng quan cấu trúc thư mục

```
SpectroChain-Dental/
├── 📊 results/                    # Kết quả đánh giá
│   ├── charts/                    # Biểu đồ trực quan
│   │   ├── *.png                  # 5 biểu đồ chính
│   │   └── visualization_charts.py # Script tạo biểu đồ
│   ├── data/                      # Dữ liệu kết quả
│   │   └── evaluation_results.json # Kết quả chi tiết
│   ├── reports/                   # Báo cáo tổng hợp
│   │   └── REAL_TIME_BENCHMARK_SUMMARY.md
│   └── README.md
│
├── 🏁 benchmarks/                 # Công cụ benchmark
│   ├── real_time/                 # Real-time benchmarking
│   │   └── benchmark_systems.py   # Core benchmark engine
│   ├── comparative/               # So sánh hệ thống
│   ├── figures/                   # Biểu đồ cũ
│   ├── results/                   # Kết quả cũ
│   └── README.md
│
├── 🎯 evaluation/                 # Công cụ đánh giá
│   ├── metrics/                   # Scripts metrics
│   │   └── performance_metrics.py # Đánh giá tổng thể
│   ├── algorithms/                # Thuật toán scoring
│   └── README.md
│
├── 🧪 tests/                      # Testing framework
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── performance/               # Performance tests
│
├── 💾 src/                        # Source code chính
│   ├── spectro_blockchain.py      # Blockchain core
│   ├── spectral_verification.py   # Verification engine
│   ├── comprehensive_evaluation.py # Legacy evaluation
│   └── evaluation_results.json    # Legacy results
│
├── 📖 docs/                       # Documentation
│   ├── technical/                 # Tài liệu kỹ thuật
│   ├── user_guide/               # Hướng dẫn người dùng
│   ├── EVALUATION_README.md       # Hướng dẫn evaluation
│   └── README.md                  # README chính
│
├── 📁 data/                       # Dữ liệu thô
│   ├── ceramic_veneer_*           # Dữ liệu ceramic
│   ├── composite_filling_*        # Dữ liệu composite
│   └── counterfeit_samples/       # Mẫu giả
│
├── ⚙️ config/                     # Configuration files
├── 🐍 venv/                       # Virtual environment
├── 📋 requirements.txt            # Dependencies
└── 📄 PROJECT_STRUCTURE.md        # File này
```

## 🎯 Workflow chính

### 1. Chạy đánh giá toàn diện
```bash
python evaluation/metrics/performance_metrics.py
```

### 2. Tạo biểu đồ
```bash
python results/charts/visualization_charts.py
```

### 3. Chạy benchmark riêng
```bash
python benchmarks/real_time/benchmark_systems.py
```

## 📊 Output Files

### 🎨 Biểu đồ (results/charts/)
1. `blockchain_performance.png` - Hiệu suất blockchain
2. `verification_accuracy.png` - Độ chính xác verification
3. `security_analysis.png` - Phân tích bảo mật STRIDE
4. `comparative_analysis.png` - So sánh hệ thống
5. `comprehensive_dashboard.png` - Dashboard tổng hợp

### 📄 Dữ liệu (results/data/)
- `evaluation_results.json` - Kết quả đánh giá đầy đủ

### 📝 Báo cáo (results/reports/)
- `REAL_TIME_BENCHMARK_SUMMARY.md` - Tóm tắt benchmark

## 🔧 Key Components

### Core Evaluation Engine
- **File**: `evaluation/metrics/performance_metrics.py`
- **Chức năng**: Đánh giá hiệu suất tổng thể
- **Output**: JSON results + console logs

### Real-time Benchmark System
- **File**: `benchmarks/real_time/benchmark_systems.py`
- **Chức năng**: Benchmark 3 hệ thống (Centralized, Blockchain-only, SpectroChain-Dental)
- **Features**: Real database, actual testing, no hardcoded values

### Visualization Engine
- **File**: `results/charts/visualization_charts.py`
- **Chức năng**: Tạo 5 biểu đồ chuyên nghiệp
- **Features**: Matplotlib, multiple chart types, professional styling

## 🏆 Key Results

| System | Overall Score | Throughput | Physical Verification |
|--------|---------------|------------|----------------------|
| **SpectroChain-Dental** | **95.94/100** | 6,431.5 TPS | **98%** |
| Blockchain Only | 51.95/100 | 0.0 TPS | 0% |
| Centralized | 31.61/100 | 22,503.99 TPS | 0% |

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run full evaluation**:
   ```bash
   python evaluation/metrics/performance_metrics.py
   ```

3. **View results**:
   - Charts: `results/charts/*.png`
   - Data: `results/data/evaluation_results.json`
   - Report: `results/reports/REAL_TIME_BENCHMARK_SUMMARY.md`

## 📈 Performance Highlights

- ✅ **100% Real-time calculation** (no hardcoded values)
- ✅ **Multi-system benchmark** (3 systems compared)
- ✅ **Security penetration testing** (STRIDE methodology)
- ✅ **Physical verification** (spectral analysis)
- ✅ **Professional visualization** (5 comprehensive charts)
- ✅ **Comprehensive scoring** (weighted metrics)

---

**🎯 Kết luận**: SpectroChain-Dental vượt trội hơn 203% so với hệ thống truyền thống dựa trên đánh giá real-time toàn diện. 