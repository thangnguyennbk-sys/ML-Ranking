# ML-Ranking — Báo cáo Đồ án 2: Chương 10 — Bài toán Phân hạng (Ranking)

**Môn học:** CSC14005 - Nhập môn Học máy  
**Học kỳ:** 2 - Năm học 2025-2026  
**Nhóm:** Group\_11

---

## Thông tin nhóm

| Họ và tên | MSSV |
|---|---|
| Nguyễn Mạnh Thắng | 23120084 |
| Trần Kim Ngọc | 23120062 |
| Nguyễn Thành Nguyên | 23120063 |
| Trần Đình Luân | 23120059 |
| Nguyễn Thiện Nhân | 23120064 |

---

## Chương sách đã chọn

**Chương 10: Ranking** — *Foundations of Machine Learning* (Mohri, Rostamizadeh & Talwalkar, 2nd ed., MIT Press, 2018)

---

## Mô tả nội dung

### Những gì đã làm

Báo cáo trình bày toàn diện Chương 10, chia thành ba phần chính:

#### Phần 1 — Lý thuyết

| Mục | Nội dung |
|---|---|
| 10.1 | Phát biểu hình thức bài toán ranking, hàm scoring, ranking loss |
| 10.2 | Generalization Bounds dựa trên Rademacher Complexity |
| 10.3 | SVM Ranking: bài toán primal, đối ngẫu, kernel trên cặp |
| 10.4 | Thuật toán RankBoost, tương đương Coordinate Descent |
| 10.5 | Bipartite Ranking, AUC, BipartiteRankBoost $\mathcal{O}(m+n)$ |
| 10.6 | Preference-based Setting, Sort-by-Degree, QuickSort ngẫu nhiên |
| 10.7 | Các tiêu chí xếp hạng: Precision, AP, MAP, DCG, NDCG |

#### Phần 2 — Thực nghiệm

Ba thực nghiệm được cài đặt **from scratch** bằng Python (không dùng scikit-learn cho phần huấn luyện):

1. **Thực nghiệm 1** — Kiểm chứng chặn sai số lý thuyết của RankBoost: so sánh sai số thực tế với $\prod Z_t$ và $\exp(-2\gamma^2 T)$.
2. **Thực nghiệm 2** — Bipartite Ranking: kiểm chứng hệ thức $R(h) + \mathrm{AUC}(h) = 1$, vẽ đường cong ROC.
3. **Thực nghiệm 3** — So sánh RankBoost và Ranking SVM (Pegasos) về tốc độ hội tụ và độ chính xác.

#### Phần 3 — Nghiên cứu tiên tiến

| Bài báo | Hội nghị |
|---|---|
| *Active Bipartite Ranking* (Cheshire et al.) | NeurIPS 2023 |
| *Which Tricks are Important for Learning to Rank?* (Lyzhin et al.) | ICML 2023 |

---

### Những điểm mở rộng so với sách

Các nội dung sau được đánh dấu `[MỞ RỘNG]` và đặt trong khung màu vàng trong báo cáo:

1. **Chứng minh đầy đủ Bổ đề Talagrand (Contraction Lemma)** bằng quy nạp toán học (giáo trình chỉ phát biểu, không chứng minh).
2. **Chứng minh tương đương RankBoost ↔ Coordinate Descent** chi tiết từng bước, bao gồm tiêu chí chọn hướng tọa độ và line search.
3. **Ví dụ phản chứng về Preference Cycle**: chỉ ra Weak Ranking Assumption bị vi phạm khi preference không bắc cầu, làm RankBoost không hội tụ.
4. **Ví dụ phản chứng về điều kiện bounded** trong Định lý tổng quát hoá (tính cần thiết của $f: \mathcal{X} \to [0,1]$).
5. **Phân tích ví dụ phản chứng về AUC**: hai mô hình có AUC bằng nhau nhưng chất lượng thực tế khác nhau hoàn toàn trong tình huống mất cân bằng dữ liệu.
6. **Chứng minh đầy đủ quan hệ gradient** $\nabla F_{\text{RankBoost}} = F^- \nabla F_{\text{AdaBoost}} + (F^+ - F^-)\nabla F^-$ trong bipartite setting.
7. **Chứng minh Exercise 10.7** (quan hệ giữa binary classification error và bipartite ranking error, cả chiều thuận và chiều ngược).
8. **Demo thực nghiệm Active Bipartite Ranking**: mô phỏng so sánh chiến lược Active vs. Passive trên dữ liệu tổng hợp với ngân sách nhãn giới hạn.
9. **Phân tích phân tích độ phức tạp tính toán** của SVM Ranking so với SVM phân loại ($\mathcal{O}(p^4)$ vs $\mathcal{O}(p^2)$).
10. **Liên hệ YetiLoss với Preference-based Ranking**: đặt câu hỏi mở về khả năng xây dựng surrogate loss kiểu YetiLoss cho bài toán không bắc cầu.

---




## Hướng dẫn tái tạo kết quả thực nghiệm

**Yêu cầu môi trường** và **Hướng dẫn chạy thực ngiệm** được trình bày trong [code/README.md](code/README.md)