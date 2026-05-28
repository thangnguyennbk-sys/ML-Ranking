# Hướng dẫn chạy Thực nghiệm & Cài đặt Mô hình (Thành viên 4)

Thư mục này chứa mã nguồn cài đặt từ đầu (from scratch) các thuật toán phân hạng (Ranking) trong Chương 10 của sách *Foundations of Machine Learning (FML)*, phục vụ cho phần **Thực nghiệm minh hoạ (Phần 2 - 30 điểm)**.

---

## 📂 Cấu trúc mã nguồn

Mã nguồn được phân tách thành các module chức năng rõ ràng để dễ dàng bảo trì và tích hợp:

*   **`requirements.txt`**: Khai báo các thư viện phụ thuộc và phiên bản đi kèm để đảm bảo tính tái tạo kết quả (reproducible).
*   **`data_generator.py`**: Trình tạo dữ liệu giả lập (synthetic data) có kiểm soát hạt giống ngẫu nhiên (`random_seed`), hỗ trợ:
    *   Tạo dữ liệu phân hạng cặp tổng quát (pairwise ranking dataset) từ hàm chấm điểm phi tuyến.
    *   Tạo dữ liệu phân hạng hai nhóm (bipartite ranking dataset) từ hai phân phối chuẩn khác nhau.
*   **`models.py`**: Triển khai hoàn toàn từ đầu (tuyệt đối không dùng thư viện ngoài cho giải thuật chính):
    *   `DecisionStump`: Bộ phân hạng yếu (weak ranker) dạng nhị phân $h: X \to \{0, 1\}$.
    *   `RankBoost`: Thuật toán boosting phân hạng cặp (Algorithm 10.1).
    *   `BipartiteRankBoost`: Thuật toán boosting phân hạng hai nhóm (Algorithm 10.2).
    *   `RankingSVM`: Mô hình SVM Ranking được tối ưu hoá trực tiếp trên hàm mất mát hinge cặp qua phương pháp hạ độ hàm dưới primal (Pegasos-like Subgradient Descent).
*   **`metrics.py`**: Chứa các hàm đánh giá chất lượng phân hạng viết từ đầu:
    *   `compute_pairwise_ranking_error`: Tính toán lỗi phân hạng cặp thực nghiệm.
    *   `compute_bipartite_ranking_error`: Tính lỗi phân hạng hai lớp.
    *   `compute_roc_curve` & `compute_auc_from_scores`: Vẽ đường cong ROC và tính chỉ số AUC.
    *   `compute_ndcg`: Đo lường mức độ tối ưu phân hạng thông qua Normalized Discounted Cumulative Gain.
*   **`run_experiments.py`**: Kịch bản chạy huấn luyện các mô hình, tự động kiểm chứng toán học các định lý/kết quả lý thuyết và xuất các đồ thị chất lượng cao (lưu vào thư mục `figures/` của code và `../report/figures/` của LaTeX).

---

## 🛠️ Hướng dẫn cài đặt & môi trường

Thực nghiệm sử dụng ngôn ngữ **Python 3.12** cùng các phiên bản thư viện sau:
*   `numpy` (phiên bản $\ge$ 1.26.2): xử lý ma trận và tính toán số học.
*   `scipy` (phiên bản $\ge$ 1.11.4): các phép so sánh ma trận nâng cao.
*   `matplotlib` (phiên bản $\ge$ 3.8.2): vẽ biểu đồ chuyên nghiệp.
*   `pandas` (phiên bản $\ge$ 2.1.4): quản lý bảng số liệu so sánh.

### Các bước cài đặt:

1.  Mở terminal tại thư mục dự án và cài đặt các gói phụ thuộc:
    ```bash
    py -m pip install -r requirements.txt
    ```

---

## 🚀 Hướng dẫn chạy thực nghiệm

Để huấn luyện toàn bộ mô hình và tạo các hình ảnh minh họa cho báo cáo LaTeX, chạy lệnh sau:

```bash
py run_experiments.py
```

Sau khi chạy xong, các tệp ảnh biểu đồ chất lượng cao sẽ tự động được sinh ra trong hai thư mục:
- `code/figures/`
- `report/figures/` (Thư mục này được thiết kế để liên kết trực tiếp với mã nguồn LaTeX `\includegraphics` trong báo cáo chính).

---

## 📊 Mô tả các thực nghiệm & Chứng minh lý thuyết

### Thực nghiệm 1: Sự hội tụ và Chặn sai số RankBoost
*   **Mục đích**: Kiểm chứng **Định lý 10.2 (FML)** chỉ ra rằng sai số huấn luyện cặp $\widehat{R}(h)$ của RankBoost giảm theo hàm mũ và bị chặn trên bởi tích các hệ số chuẩn hóa $\prod_{s=1}^t Z_s$:
    $$\widehat{R}(g_t) \le \prod_{s=1}^t Z_s$$
*   **Kết quả**: Đồ thị `experiment1_rankboost_bounds.png` hiển thị dưới thang đo Logarit cho thấy rõ đường sai số thực tế luôn nằm nghiêm ngặt dưới đường chặn lý thuyết và giảm dần về 0.

### Thực nghiệm 2: Phân hạng nhị phân (Bipartite Ranking) và AUC
*   **Mục đích**: Chứng minh mối liên hệ giữa hàm mục tiêu tối ưu của bài toán phân hạng nhị phân và Area under the ROC Curve:
    $$R(h) = 1 - \mathrm{AUC}_{book}(h)$$
*   **Kết quả**: Đồ thị `experiment2_bipartite_roc_scores.png` trực quan hoá:
    1.  Đường cong ROC và tính chỉ số AUC trên tập test.
    2.  Sự phân tách điểm số dự báo $g(x)$ của mô hình giữa hai tập mẫu dương ($X^+$) và mẫu âm ($X^-$).
    3.  In ra màn hình kết quả kiểm chứng toán học: $R(h) + \mathrm{AUC}_{book} = 1.0$, khẳng định tính đúng đắn của định nghĩa lý thuyết.

### Thực nghiệm 3: So sánh hiệu năng giữa RankBoost và Ranking SVM
*   **Mục đích**: So sánh trực tiếp chất lượng phân hạng (sai số cặp trên tập huấn luyện/kiểm thử) và thời gian thực thi của thuật toán Boosting so với SVM.
*   **Kết quả**: Đồ thị `experiment3_model_comparison.png` vẽ đường cong hội tụ sai số của hai mô hình và so sánh trực quan sai số thực tế dưới dạng biểu đồ cột.
