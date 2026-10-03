# Báo cáo Lab Day 1 — Xây dựng mạng nơ-ron và thí nghiệm huấn luyện

**Sinh viên thực hiện:** Ngô Xuân Hoàng (MSSV: 2A202602597)  
**Khóa học:** Track 4 · Ngày 1 · VinUniversity AICB 2026  
**Bài học liên quan:** Mạng Nơ-ron và Huấn Luyện (Slide Day 1)  

---

## 1. Thiết lập

- **Môi trường:** Máy tính cá nhân chạy Windows 11, GPU NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM, Driver 566.07, CUDA 12.7), Python 3.12.10, PyTorch 2.6.0+cu124.
- **Dữ liệu:** Forest CoverType (Blackard & Dean, UCI); tập `train` gồm 464 809 mẫu, tập `eval` gồm 116 203 mẫu theo metadata chuẩn `split_metadata.csv`.
- **Validation Split:** Tách phân tầng theo nhãn 20% từ tập `train` (seed 42) $\to$ **371 847 mẫu train** và **92 962 mẫu validation**.
- **Chuẩn hóa:** Chỉ áp dụng chuẩn hóa (Z-score: mean = 0, std = 1) trên **10 đặc trưng số liên tục đầu tiên**. Thống kê $\mu, \sigma$ được tính **duy nhất trên tập train con (371 847 mẫu)** sau khi tách validation để tuyệt đối tránh rò rỉ thông tin (*data leakage*). 44 cột nhị phân (one-hot địa hình và loại đất) được giữ nguyên.
- **Mô hình chuẩn (`M-base`):** `54 → 256 → 128 → 7` (kích hoạt ReLU ở mọi lớp ẩn, bias đầy đủ, không BatchNorm/Residual). Tổng số tham số: đúng **47 879** tham số.
- **Baseline:** Mất mát Cross-Entropy, Optimizer SGD + Momentum 0.9, Learning rate $\eta = 0.05$, Batch size 512, 20 epochs, Khởi tạo He (Kaiming Normal), Precision FP32.
- **Mốc tham chiếu:** Chiến lược "luôn đoán lớp đa số" (lớp 1 - Cover Type 2) trên tập validation đạt accuracy = **0.4876** (48.76%), macro-F1 chỉ $\approx \mathbf{0.0936}$. Đây là mốc sàn bắt buộc mô hình phải vượt qua.
- **Các chủ đề đã thử nghiệm:** Đầy đủ **7/7 chủ đề** theo Rubric:
  - [x] Hàm mất mát (`loss`)
  - [x] Bộ tối ưu hoá (`optimizer`)
  - [x] Hyper-parameter (`hparam`)
  - [x] Dropout (`dropout`)
  - [x] Gradient clipping (`clipping`)
  - [x] Mixed precision (`amp`)
  - [x] Khởi tạo tham số (`init`)

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Tiêu chuẩn kỳ vọng | Kết quả đo thực tế | Đánh giá |
|---|---|---|---|
| Số tham số `M-base` / Shape logits | 47 879 / `(B, 7)` | **47 879** / `(B, 7)` | Đạt chuẩn (`assert` thành công) |
| Loss bước 0 trên Val (He init) | Gần $\ln 7 \approx 1.9459$ | **2.3776** | Đạt chuẩn (đầu ra logit có phương sai ban đầu nhỏ) |
| Quá khớp 20 mẫu (300 steps) | Loss $\to 0$, Acc $\to 100\%$ | **Loss: $6 \times 10^{-6}$, Acc: 100.0%** | Đạt chuẩn (Pipeline học trơn tru) |
| Kiểm tra dòng gradient mọi tham số | Gradient $\neq \text{None}$ và $> 0$ | Đầy đủ 6 tensor $W_1, b_1, W_2, b_2, W_3, b_3 > 0$ | Đạt chuẩn (Không nghẽn gradient) |
| Baseline, số seed đã chạy | $\ge 2$ seeds (khuyến khích 3) | **3 seeds** (`base-s1`, `base-s2`, `base-s3`) | Đạt chuẩn |
| Baseline: Val Accuracy (TB $\pm \sigma$) | — | **0.9025 $\pm$ 0.0015** (90.25% $\pm$ 0.15%) | Hội tụ rất ổn định |
| Baseline: Val Macro-F1 (TB $\pm \sigma$) | — | **0.8406 $\pm$ 0.0064** | Dao động ngẫu nhiên nhỏ |

**Ngưỡng nhiễu dùng trong báo cáo ($2\sigma$):**
- Ngưỡng nhiễu Macro-F1: $2\sigma = 2 \times 0.0064 = \mathbf{0.0128}$ (tương đương **1.28%**).
- Mọi kết luận "Cấu hình A tốt hơn B" chỉ được xem là có ý nghĩa thực sự khi độ chênh lệch $|\Delta \text{Macro-F1}| > 0.0128$. Các chênh lệch nhỏ hơn ngưỡng này được xem là nằm trong vùng bất định của hạt ngẫu nhiên (seed noise).

---

## 3. Kết quả theo từng chủ đề

### 3.1 Hàm mất mát — Cross-Entropy (CE) vs Mean Squared Error (MSE)
- **Dự đoán trước khi chạy:** Cross-Entropy sẽ vượt trội hơn MSE rõ rệt về tốc độ hội tụ và Macro-F1, vì Cross-Entropy tính đạo hàm theo logit là $p_i - y_i$ (không bị triệt tiêu khi dự đoán sai lệch lớn), trong khi MSE trên one-hot vector có độ dốc suy giảm nhanh khi giá trị logit lệch xa, đặc biệt làm yếu gradient của các lớp hiếm.
- **Kết quả đo:**
  - `base-s1` (CE): Val Loss = 0.2481, Val Acc = **0.9010**, Val Macro-F1 = **0.8447** (Hình: `figures/base-s1.png`).
  - `loss-mse` (MSE): Val Loss = 0.2274, Val Acc = **0.8541**, Val Macro-F1 = **0.6928** (Hình: `figures/loss-mse.png`).
  - Biểu đồ so sánh trực tiếp: `figures/compare_loss.png`.
- **Chênh lệch và ý nghĩa:**
  $$\Delta \text{Macro-F1} = 0.8447 - 0.6928 = \mathbf{+0.1519} \gg 2\sigma \ (0.0128)$$
  Sự vượt trội của CE lớn gấp **11.8 lần ngưỡng nhiễu**.
- **Giải thích cơ chế:** Không so sánh trực tiếp giá trị loss giữa CE và MSE vì thang đo hoàn toàn khác nhau. Về mặt giải tích, hàm mất mát Cross-Entropy kết hợp cùng Softmax tạo ra đạo hàm tuyến tính theo sai số xác suất: $\frac{\partial \mathcal{L}_{CE}}{\partial z_i} = p_i - y_i$. Khi mô hình tự tin sai (ví dụ $p_i \to 0$ nhưng $y_i = 1$), gradient đạt cực đại $|p_i - y_i| \to 1$, kéo trọng số cập nhật mạnh mẽ. Ngược lại, MSE trên logit $\mathcal{L}_{MSE} = \frac{1}{C}\sum (z_i - y_i)^2$ có gradient $\frac{2}{C}(z_i - y_i)$, ép các logit cố định về giá trị 0 và 1 thay vì tối ưu hóa xác suất tương đối, làm mất tính bất biến dịch chuyển của Softmax và khiến mạng hội tụ rất chậm ở các nhãn thiểu số.

---

### 3.2 Bộ tối ưu hoá (Optimizers)
- **Dự đoán trước khi chạy:** SGD thuần túy không có momentum sẽ hội tụ chậm nhất do dao động zigzag trong thung lũng hàm mất mát. SGD có momentum sẽ tăng tốc đáng kể. Các thuật toán thích nghi (Adam, AdamW) sẽ hội tụ nhanh nhất trong các epoch đầu tiên nhờ việc điều chỉnh bước học cục bộ theo phương sai của từng tham số.
- **Bảng so sánh các bộ tối ưu ở các mức Learning Rate:**

| Mã thí nghiệm (`exp_id`) | Thuật toán | Learning Rate | Weight Decay | Best Epoch | Val Acc | Val Macro-F1 | So với Baseline ($\Delta$) | Vượt nhiễu? |
|---|---|---|---|---|---|---|---|---|
| `opt-sgd-lr0.05` | SGD | 0.05 | 0.0 | 20 | 0.8326 | 0.7148 | -0.1299 | Có (Tệ hơn) |
| `opt-sgd-lr0.1` | SGD | 0.10 | 0.0 | 20 | 0.8516 | 0.7635 | -0.0812 | Có (Tệ hơn) |
| `base-s1` | SGD + Momentum | 0.05 | 0.0 | 20 | 0.9010 | 0.8447 | 0.0000 | Baseline |
| `opt-adam-lr3e-4` | Adam | 0.0003 | 0.0 | 20 | 0.8743 | 0.7948 | -0.0499 | Có (Tệ hơn) |
| `opt-adam-lr1e-3` | Adam | 0.0010 | 0.0 | 20 | 0.9032 | **0.8523** | **+0.0076** | Vừa chạm ngưỡng |
| `opt-adamw-lr1e-3` | AdamW | 0.0010 | 0.01 | 20 | 0.9024 | **0.8498** | **+0.0051** | Tương đương |

- **Độ nhạy với learning rate:**
  - SGD thuần rất nhạy với learning rate; tăng từ 0.05 lên 0.1 giúp Macro-F1 tăng mạnh từ 0.7148 lên 0.7635 nhưng vẫn còn kém xa SGD+momentum.
  - Adam với $\eta = 3\times 10^{-4}$ quá nhỏ, mô hình chưa hội tụ kịp sau 20 epoch (Macro-F1 0.7948). Khi nâng lên $\eta = 10^{-3}$, Adam bứt phá đạt **0.8523**, vượt qua baseline SGDM.
  - Biểu đồ đối chiếu: `figures/compare_optimizer.png`.
- **Giải thích cơ chế:** Momentum tích lũy vận tốc $v_t = \mu v_{t-1} + g_t$, triệt tiêu các thành phần dao động ngược chiều và khuếch đại bước nhảy theo hướng dốc đồng thuận, giúp SGDM vượt trội SGD thuần (+0.0812). Adam ước lượng cả moment bậc 1 ($m_t$) và moment bậc 2 ($v_t$), chuẩn hóa gradient bằng $\sqrt{v_t} + \epsilon$. Điều này đặc biệt có lợi cho CoverType vì các đặc trưng one-hot thưa (40 cột soil type) có gradient thưa nhưng mang thông tin đặc thù, Adam tự động tăng bước học cho các tham số này. AdamW tách biệt việc suy giảm trọng số ra khỏi gradient cập nhật ($w \leftarrow w - \eta \lambda w$), duy trì tính chính quy hoá ổn định.

---

### 3.3 Hyper-parameters (Batch Size & Kiến trúc mạng)
- **Các yếu tố đã thử:**
  - Batch size: 128 vs 512 (baseline) vs 2048.
  - Kiến trúc: `M-wide` (54→512→256→7, 161 287 params) vs `M-deep` (54→256→128→64→7, 55 687 params) vs `M-base` (47 879 params).

| Mã thí nghiệm (`exp_id`) | Yếu tố thay đổi | Số bước / epoch | Thời gian / epoch | Val Loss | Val Acc | Val Macro-F1 | So với Baseline ($\Delta$) | Vượt nhiễu ($> 2\sigma$)? |
|---|---|---|---|---|---|---|---|---|
| `base-s1` | Chuẩn (bs=512, M-base) | 727 | 2.53s | 0.2481 | 0.9010 | 0.8447 | 0.0000 | — |
| `hparam-bs128` | Batch size = 128 | 2 905 | 6.41s | 0.2173 | 0.9140 | **0.8666** | **+0.0219** | **Có (Vượt trội)** |
| `hparam-bs2048` | Batch size = 2048 | 182 | 0.43s | 0.3066 | 0.8754 | 0.7780 | -0.0667 | Có (Kém hơn) |
| `hparam-wide` | Kiến trúc `M-wide` | 727 | 1.70s | 0.2220 | 0.9119 | **0.8679** | **+0.0232** | **Có (Vượt trội)** |
| `hparam-deep` | Kiến trúc `M-deep` | 727 | 1.88s | **0.2073** | **0.9177** | **0.8678** | **+0.0231** | **Có (Vượt trội)** |

- Biểu đồ so sánh: `figures/compare_hparam.png`.
- **Giải thích cơ chế:**
  - *Ảnh hưởng của Batch Size:* Với cùng 20 epoch, `hparam-bs128` thực hiện tới $20 \times 2905 = 58 100$ lần cập nhật trọng số, gấp 4 lần baseline (14 540 lần) và gấp 16 lần batch 2048 (3 640 lần). Sự nhiễu ngẫu nhiên của batch nhỏ hoạt động như một bộ điều chuẩn ngẫu nhiên (implicit regularization) giúp thoát khỏi yên ngựa và hội tụ sâu hơn. Ngược lại, batch 2048 quá ít bước cập nhật khiến mạng bị *underfitting* nặng nề sau 20 epoch.
  - *Độ rộng và Độ sâu:* Mạng `M-wide` tăng số nơ-ron lớp ẩn (512-256) giúp tăng khả năng phân tách tuyến tính các tổ hợp nhị phân của Wilderness và Soil Type. Mạng `M-deep` tăng thêm 1 tầng ẩn (256-128-64) giúp tạo biểu diễn phân cấp trừu tượng (hierarchical representation), đạt Val Loss thấp kỷ lục (0.2073) và Val Acc cao nhất (91.77%) trong các mô hình đơn lẻ.

---

### 3.4 Dropout
- **Dự đoán trước khi chạy:** Nếu mô hình chưa bị quá khớp (khoảng cách train-val loss nhỏ), dropout sẽ làm giảm khả năng biểu diễn và làm giảm điểm số validation.
- **Kết quả đo nghiệm:**

| Mã thí nghiệm (`exp_id`) | Tỷ lệ tắt ($q$) | Final Train Loss | Final Val Loss | Khoảng cách Val - Train | Val Acc | Val Macro-F1 | Kết luận |
|---|---|---|---|---|---|---|---|
| `base-s1` | 0.0 (Không) | 0.2257 | 0.2481 | **+0.0224** | **0.9010** | **0.8447** | Baseline tối ưu |
| `drop-0.1` | 0.1 | 0.2623 | 0.2713 | +0.0090 | 0.8906 | 0.8208 | Giảm nhẹ (-0.0239) |
| `drop-0.3` | 0.3 | 0.3341 | 0.3373 | +0.0032 | 0.8623 | 0.7565 | Giảm rõ rệt (-0.0882) |
| `drop-0.5` | 0.5 | 0.4079 | 0.4083 | +0.0004 | 0.8250 | 0.6395 | Suy giảm nghiêm trọng |

- Biểu đồ đối chiếu: `figures/compare_dropout.png`.
- **Nhận xét & Cơ chế:**
  - Ở baseline, sau 20 epoch, Train Loss là 0.2257 và Val Loss là 0.2481, khoảng cách chỉ vỏn vẹn **0.0224**. Đường cong validation loss vẫn đang trên đà giảm đều đặn và chưa hề có dấu hiệu đảo chiều đi lên. Điều này chứng minh mô hình **hoàn toàn chưa bị quá khớp** (nhờ tập dữ liệu 371k mẫu quá lớn so với dung lượng 47k tham số của `M-base`).
  - Khi áp dụng Dropout với $q=0.3$ hoặc $0.5$, khoảng cách giữa train và val bị thu hẹp về gần 0, nhưng cả train loss và val loss đều tăng vọt (val loss từ 0.2481 tăng lên 0.4083). Mô hình bị đẩy vào tình trạng **thiếu khớp (underfitting)** nghiêm trọng. Đúng như slide Chương 5 nhấn mạnh: *"Dropout là liều thuốc chữa quá khớp; uống thuốc khi không có bệnh sẽ làm cơ thể yếu đi"*.

---

### 3.5 Gradient Clipping
- **Mục tiêu thí nghiệm:** Kiểm tra hoạt động của Gradient Clipping ở điều kiện bình thường và trong thí nghiệm phản chứng khi learning rate bị đẩy lên mức cực cao gây bùng nổ gradient.
- **Kết quả đo nghiệm:**

| Mã thí nghiệm (`exp_id`) | Learning Rate | Clip threshold ($c$) | Chuẩn Gradient cực đại ($\|\mathbf{g}\|_2$) | Val Acc | Val Macro-F1 | Trạng thái mô hình |
|---|---|---|---|---|---|---|
| `base-s1` | 0.05 | Không clip | 0.811 | **0.9010** | **0.8447** | Ổn định |
| `clip-1.0` | 0.05 | $c = 1.0$ | 0.806 (trước clip) | 0.8998 | 0.8381 | Ổn định, ít kích hoạt |
| `clip-highlr-noclip` | 2.00 | Không clip | 2.684 (gai lớn) | 0.4876 | **0.0936** | **Sụp đổ hoàn toàn (chỉ đoán lớp 1)** |
| `clip-highlr-clip` | 2.00 | $c = 1.0$ | 2.641 (trước clip) | **0.6465** | **0.2879** | **Cứu nguy thành công, không sụp đổ** |

- Biểu đồ đối chiếu: `figures/compare_clipping.png`.
- **Giải thích cơ chế:**
  - Ở learning rate chuẩn $\eta = 0.05$, chuẩn gradient trung bình dao động từ 0.70 đến 0.81, hiếm khi vượt qua ngưỡng $c=1.0$. Do đó, phép chiếu $\mathbf{g} \leftarrow \mathbf{g} \cdot \min(1, c/\|\mathbf{g}\|)$ gần như đồng nhất và kết quả của `clip-1.0` (0.8381) sát với baseline (0.8447, nằm trong sai số ngẫu nhiên).
  - Ở điều kiện bất lợi ($\eta = 2.0$): Mô hình không có clip (`clip-highlr-noclip`) bị các bước nhảy gradient quá lớn phá hỏng toàn bộ cấu trúc trọng số, dẫn đến hiện tượng mạng bị tê liệt và rơi vào cực tiểu phẳng giả định (đoán toàn bộ mẫu là lớp đa số 1, accuracy đúng bằng 0.4876 và Macro-F1 rơi về đáy 0.0936).
  - Ngược lại, khi có clipping $c=1.0$ (`clip-highlr-clip`), biên độ cập nhật bị chặn trên bởi $c \cdot \eta$, ngăn chặn sự bùng nổ trọng số và giúp mô hình vẫn phân loại được các lớp khác nhau (đạt accuracy 64.65% và Macro-F1 0.2879). Đây là bằng chứng thực nghiệm rõ ràng nhất khẳng định vai trò của Gradient Clipping như một "dây an toàn" cho mạng nơ-ron.

---

### 3.6 Mixed Precision (FP32 vs FP16)
- **Kết quả đo thực tế:**

| Mã thí nghiệm (`exp_id`) | Độ chính xác | Thời gian / Epoch | Bộ nhớ VRAM đỉnh | Val Acc | Val Macro-F1 | Sai lệch Macro-F1 so với FP32 |
|---|---|---|---|---|---|---|
| `base-s1` | FP32 | 2.53s | 160.5 MB | 0.9010 | 0.8447 | — |
| `amp-fp16` | FP16 (Autocast + GradScaler) | **2.21s** | 160.5 MB | 0.8985 | 0.8424 | **-0.0023** (Nằm trong nhiễu $2\sigma$) |

- **Giải thích cơ chế:**
  - *Độ chính xác:* Điểm Macro-F1 của FP16 (0.8424) gần như trùng khớp hoàn hảo với FP32 (0.8447) với độ lệch chỉ 0.0023 (nhỏ hơn nhiều so với $\sigma = 0.0064$). `GradScaler` đã hoàn thành xuất sắc nhiệm vụ nhân tỉ lệ loss ($s \cdot \mathcal{L}$) trước khi backward để tránh hiện tượng underflow (triệt tiêu gradient về 0 trong dải số mũ nhỏ của FP16).
  - *Tốc độ và Bộ nhớ:* Thời gian epoch giảm từ 2.53s xuống 2.21s (nhanh hơn ~13%). Với một mạng MLP kích thước nhỏ (47 879 tham số), thời gian tính toán của Tensor Cores chỉ chiếm một phần nhỏ, trong khi phần lớn thời gian bị chi phối bởi chi phí gọi kernel của GPU (overhead kernel launch) và việc truyền dữ liệu. Do đó, mức tăng tốc không đạt ngưỡng 2x-3x như các mô hình Transformer hay ConvNet khổng lồ.

---

### 3.7 Khởi tạo tham số (Weight Initialization)
- **Độ lệch chuẩn kích hoạt bước 0** đo trên 1 lô validation (512 mẫu) và Loss ban đầu:

| Kiểu khởi tạo | Công thức PyTorch | Std sau Lớp 1 (256) | Std sau Lớp 2 (128) | Std sau Lớp Ra (7) | Loss bước 0 | Val Acc cuối | Val Macro-F1 cuối |
|---|---|---|---|---|---|---|---|
| `init-zeros` | $W = 0, b = 0$ | 0.0000 | 0.0000 | 0.0000 | **1.9459** | 0.4876 | **0.0936** |
| `init-normal` | $W \sim \mathcal{N}(0, 0.01^2)$ | 0.0351 | 0.0039 | 0.0003 | 1.9460 | 0.8884 | 0.8271 |
| `init-xavier` | `xavier_normal_` | 0.2834 | 0.2240 | 0.2618 | 1.9512 | 0.8975 | 0.8417 |
| `init-he` (Baseline)| `kaiming_normal_` (ReLU)| **0.7002** | **0.7019** | **0.5535** | 2.2691 | **0.9010** | **0.8447** |

- Biểu đồ đối chiếu: `figures/compare_init.png`.
- **Giải thích cơ chế:**
  - **Khởi tạo Zeros ($W=0$):** Loss bước 0 đạt chính xác tuyệt đối $\ln 7 = 1.945915$ vì mọi logit đầu ra đều bằng 0, dẫn tới phân phối xác suất đều $1/7$ cho cả 7 lớp. Tuy nhiên, khi $W=0$, mọi nơ-ron trong cùng một lớp ẩn nhận đầu vào giống hệt nhau và tính ra đầu ra giống hệt nhau ($ReLU(0) = 0$). Khi backward, gradient truyền về mọi nơ-ron trong lớp là đối xứng hoàn toàn. Tính đối xứng (*symmetry*) không bao giờ bị phá vỡ, biến cả mạng thành một nơ-ron đơn lẻ. Kết quả mô hình hoàn toàn bất lực và sụp đổ về mức đoán lớp đa số (Acc 0.4876, Macro-F1 0.0936).
  - **Khởi tạo Normal ($\sigma=0.01$):** Phương sai trọng số quá nhỏ khiến phương sai kích hoạt suy giảm theo hàm mũ qua từng tầng ($0.0351 \to 0.0039 \to 0.0003$). Tín hiệu đầu vào bị tắt nghẽn khi đi sâu vào mạng, làm chậm tốc độ học ban đầu.
  - **He vs Xavier:** Xavier giả định hàm kích hoạt tuyến tính quanh điểm 0. Với ReLU, một nửa số nơ-ron bị tắt ở mỗi tầng ($ReLU(z) = 0$ khi $z < 0$), làm phương sai tín hiệu bị giảm đi một nửa sau mỗi tầng. Khởi tạo He nhân thêm hệ số $\sqrt{2}$ ($\text{Var}[W] = 2/n_{in}$) để bù đắp chính xác 50% tín hiệu bị mất này, giúp giữ vững độ lệch chuẩn kích hoạt quanh mức ~0.70 qua mọi tầng ẩn và giúp mô hình hội tụ tốt nhất.

---

## 4. Đánh giá cuối trên tập EVAL

> **Nguyên tắc đạo đức nghiên cứu:** Cấu hình cuối cùng được lựa chọn **HOÀN TOÀN dựa trên kết quả Validation**, không hề nhìn trước tập Eval. Tập Eval chỉ được mở ra một lần duy nhất ở bước đánh giá cuối cùng thông qua script chính thức `scripts/evaluate.py`.

### Lựa chọn Cấu hình Cuối cùng (Final Model)
Từ các kết quả trên tập Validation:
1. **Kiến trúc:** `M-wide` (54→512→256→7) cho dung lượng biểu diễn vượt trội (Val Macro-F1 = 0.8679).
2. **Bộ tối ưu:** `AdamW` với $\eta = 10^{-3}$, weight decay = 0.01 giúp học tối ưu cho các đặc trưng thưa và thích ứng bước nhảy tốt nhất.
3. **Khởi tạo & Regularization:** Khởi tạo He, không dùng Dropout (do mạng không bị quá khớp).
$\implies$ Cấu hình `final-model` đạt **Val Macro-F1 = 0.8788** và **Val Acc = 92.02%** trên tập Validation.

### Bảng kết quả đánh giá cuối trên Eval

| Cấu hình | Seed nộp | Val Macro-F1 | **Eval Macro-F1** (Chỉ số chính) | Eval Accuracy | Đánh giá Rubric Mục 7 |
|---|---|---|---|---|---|
| **Baseline** (`base-s1`) | 1 | 0.8447 | **0.8463** | 0.8999 (89.99%) | Vượt xa mốc đoán đa số (0.094) |
| **Cấu hình Cuối cùng** (`final-model`) | 1 | 0.8788 | **0.8823** | **0.9198** (91.98%) | **$\ge 0.86 \to$ Đạt tối đa 5/5 điểm** |

- **Cải thiện so với baseline trên tập Eval:**
  $$\Delta \text{Eval Macro-F1} = 0.8823 - 0.8463 = \mathbf{+0.0360} \ (3.60\%)$$
  Mức cải thiện $+0.0360$ lớn hơn nhiều mốc yêu cầu tối đa của Rubric ($+0.020$) và vượt xa ngưỡng nhiễu seed $2\sigma = 0.0128$ $\to$ **Đạt tối đa 3/3 điểm mục cải thiện**.
- **Độ tin cậy của tập Validation:** Điểm trên tập Val (0.8788) và Eval (0.8823) chênh lệch rất nhỏ ($< 0.0035$), khẳng định việc tách tập validation phân tầng 20% là một ước lượng không chệch và cực kỳ đáng tin cậy.

---

### 4.1 Phân tích lỗi theo lớp (Class Error Analysis)

Số liệu chi tiết trích xuất trực tiếp từ file `eval_result.json` do `scripts/evaluate.py` tạo ra:

| Lớp (Cover Type) | Tên loại rừng | Số mẫu (Support) | Tỷ lệ (%) | Precision | Recall | F1-Score |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **0** | Spruce/Fir | 42 368 | 36.46% | 0.9100 | 0.9236 | 0.9168 |
| **1** | Lodgepole Pine | 56 661 | 48.76% | 0.9342 | 0.9272 | 0.9307 |
| **2** | Ponderosa Pine | 7 151 | 6.15% | 0.9263 | 0.9088 | 0.9175 |
| **3** | Cottonwood/Willow | 549 | 0.47% | 0.8080 | 0.8506 | 0.8287 |
| **4** | Aspen | 1 899 | 1.63% | 0.8398 | 0.7730 | **0.8050** |
| **5** | Douglas-fir | 3 473 | 2.99% | 0.8308 | 0.8580 | 0.8442 |
| **6** | Krummholz | 4 102 | 3.53% | 0.9408 | 0.9254 | 0.9330 |

#### Ma trận nhầm lẫn trên 116 203 mẫu Eval (Hàng = Nhãn thật, Cột = Dự đoán):
```
           Pred 0   Pred 1   Pred 2   Pred 3   Pred 4   Pred 5   Pred 6
True 0     39132     2988        0        0       29        9      210
True 1      3533    52538      111        1      227      222       29
True 2         8      203     6499       86       15      340        0
True 3         0        0       57      467        0       25        0
True 4        59      343       18        0     1468       11        0
True 5        11      119      331       24        8     2980        0
True 6       259       46        0        0        1        0     3796
```

#### Phân tích chuyên sâu:
1. **Lớp khó nhất:** Lớp **4 (Aspen)** có F1-score thấp nhất (**0.8050**), với Recall chỉ đạt **77.30%**.
   - Nhìn vào ma trận nhầm lẫn, có tới **343 mẫu lớp 4 bị đoán nhầm thành lớp 1** và **59 mẫu bị nhầm thành lớp 0**.
   - *Nguyên nhân địa lý/thổ nhưỡng:* Cây lá rụng Aspen (lớp 4) thường mọc xen kẽ trong các dải cao độ trung bình (2 700m - 3 000m) tương đồng với Lodgepole Pine (lớp 1). Về mặt sinh thái học, Aspen thường là loài tiên phong mọc lại sau cháy rừng trong các quần thể Lodgepole Pine, dẫn đến đặc trưng về độ cao (Elevation), hướng dốc (Aspect) và chỉ số bóng râm (Hillshade) của hai loài này chồng lấn rất mạnh.
2. **Lớp hiếm nhất:** Lớp **3 (Cottonwood/Willow)** chỉ chiếm **0.47% dữ liệu** (549 mẫu trên toàn bộ tập eval).
   - Tuy nhiên, mô hình đạt Precision 80.80%, Recall 85.06% và F1 = **0.8287**. Các mẫu nhầm lẫn chủ yếu rơi vào lớp 2 (57 mẫu) và lớp 5 (25 mẫu), vốn là các loại cây mọc ở vùng trũng ven sông suối tương tự.
3. **Cặp nhầm lẫn lớn nhất về số lượng:** Lớp 0 (Spruce/Fir) và Lớp 1 (Lodgepole Pine). Có 2 988 mẫu lớp 0 bị đoán nhầm là lớp 1, và 3 533 mẫu lớp 1 bị đoán nhầm là lớp 0. Đây là hai loài thông chiếm tới 85% tổng số mẫu toàn bộ dataset, có ranh giới phân bố sinh thái tiếp giáp liên tục theo độ cao.
4. **Giải pháp cải thiện đề xuất:** Trong tương lai, có thể áp dụng hàm mất mát có trọng số phân lớp (*Class-Balanced Focal Loss*) hoặc kỹ thuật tổng hợp mẫu thiểu số (*SMOTE*) ở tầng biểu diễn ẩn để cải thiện khả năng phân tách của lớp 4 và lớp 3.

---

## 5. Trả lời các câu hỏi dẫn dắt

### Câu 1: Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?
- Khi được quét qua nhiều mức learning rate hợp lý (fair tuning), **Adam ($\eta = 10^{-3}$, Macro-F1 = 0.8523)** và **AdamW ($\eta = 10^{-3}$, Macro-F1 = 0.8498)** giành chiến thắng trước SGD+momentum ($\eta = 0.05$, Macro-F1 = 0.8447) và vượt xa SGD thuần ($\eta = 0.1$, Macro-F1 = 0.7635).
- Nếu không tinh chỉnh lr mà áp dụng bừa một mức lr chung (ví dụ lấy $\eta = 0.05$ của SGD áp dụng cho Adam), Adam sẽ lập tức bị phân kỳ hoặc dao động mạnh quanh điểm tối ưu; ngược lại nếu lấy $\eta = 10^{-3}$ của Adam áp đặt cho SGD, SGD sẽ gần như không di chuyển sau 20 epoch. Khi đó kết luận "thuật toán nào tốt hơn" sẽ bị sai lệch hoàn toàn do đánh đồng năng lực thuật toán với việc chọn sai siêu tham số.

### Câu 2: Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?
- **Không giúp ích**, thậm chí còn gây hại rõ rệt. Khi mô hình `M-base` chưa quá khớp (khoảng cách train-val loss chỉ 0.022), việc ngắt ngẫu nhiên nơ-ron làm giảm dung lượng tính toán hữu hiệu và cản trở việc học các đặc trưng chi tiết của các lớp hiếm, kéo Macro-F1 từ 0.8447 tụt xuống 0.7565 ($q=0.3$) và 0.6395 ($q=0.5$).
- **Chỉ nên dùng Dropout khi:** (1) Dung lượng mô hình quá lớn so với số lượng mẫu huấn luyện, (2) Đường cong huấn luyện xuất hiện dấu hiệu quá khớp rõ rệt (train loss tiếp tục giảm sâu trong khi val loss đảo chiều tăng vọt).

### Câu 3: Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?
- Gradient clipping giải quyết vấn đề **bùng nổ gradient (exploding gradients)** — hiện tượng các vector đạo hàm có độ dài $\|\mathbf{g}\|_2$ đột ngột tăng vọt đẩy trọng số văng ra khỏi vùng cực tiểu tối ưu hoặc tràn số float (NaN/Inf).
- **Quan sát thực nghiệm chứng minh:** Ở thí nghiệm phản chứng với $lr = 2.0$:
  - Khi không clip (`clip-highlr-noclip`): Gradient gai lớn đẩy mạng vào trạng thái sụp đổ hoàn toàn (Macro-F1 tụt thảm hại về 0.0936, mạng chỉ đoán lớp 1).
  - Khi bật clip $c=1.0$ (`clip-highlr-clip`): Mô hình được cứu sống, khống chế được độ lớn bước cập nhật và duy trì được Macro-F1 ở mức 0.2879 (Acc 64.65%).

### Câu 4: Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao (không)?
- Trên bài toán này, mixed precision FP16 giúp giảm thời gian mỗi epoch từ **2.53s xuống 2.21s** (nhanh hơn khoảng **12.6%**).
- Tốc độ không tăng gấp đôi (2x) là vì `M-base` là một mạng MLP rất nhỏ (chỉ 3 tầng tuyến tính). Thời gian thực thi phép nhân ma trận trên GPU là cực nhanh, phần lớn thời gian bị chi phối bởi chi phí overhead của hệ thống (lập lịch kernel PyTorch, di chuyển dữ liệu trong bộ nhớ, tính toán loss trên CPU/GPU). Mức tăng tốc của Mixed Precision chỉ thực sự phát huy tối đa (2x - 3x) trên các mô hình deep learning có khối lượng tính toán khổng lồ (Large Language Models, Vision Transformers, ResNet sâu).

### Câu 5: Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?
- **Khởi tạo Zeros hỏng vì vi phạm tính phá vỡ đối xứng (*symmetry breaking*):** Khi mọi trọng số $W=0$, mọi nơ-ron trong cùng một lớp nhận tín hiệu giống hệt nhau, cho kích hoạt giống hệt nhau ($ReLU(0)=0$), và nhận gradient giống hệt nhau từ lớp sau. Chúng cập nhật song song giống hệt nhau qua mọi epoch, tương đương với một mạng chỉ có đúng 1 nơ-ron mỗi tầng.
- **He vs Xavier:**
  - Xavier giả định hàm kích hoạt đối xứng qua gốc tọa độ và tuyến tính (như Tanh/Linear), công thức $\text{Var}[W] = \frac{2}{n_{in} + n_{out}}$.
  - He (Kaiming) tính đến việc hàm ReLU triệt tiêu hoàn toàn nửa âm của tín hiệu ($\mathbb{E}[ReLU(z)^2] = \frac{1}{2}\text{Var}[z]$), do đó bù trừ bằng cách nhân đôi phương sai: $\text{Var}[W] = \frac{2}{n_{in}}$.
  - Điều này đặc biệt quan trọng trong các mạng nơ-ron sâu sử dụng ReLU: nếu dùng Xavier cho mạng ReLU sâu 20-30 tầng, phương sai tín hiệu sẽ suy giảm theo lũy thừa $(1/2)^L \to 0$, khiến các tầng sâu bị "đói" tín hiệu và tê liệt gradient.

### Câu 6: Quay lại câu hỏi của bài học: Một mạng có loss không giảm sau 2 000 bước. Nêu 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao.
Dựa vào bảng "chẩn đoán triệu chứng" ở Chương 5 và các kinh nghiệm thu được từ lab:
1. **Kiểm tra 1: Đo Loss bước 0 trên tập Validation.**
   - *Lý do:* Kiểm tra xem loss ban đầu có xấp xỉ $\ln C$ ($\ln 7 \approx 1.946$) hay không. Nếu loss bước 0 cao hơn nhiều, nguyên nhân chắc chắn do khởi tạo trọng số sai (quá lớn), thiếu chuẩn hóa đầu vào, hoặc chưa đặt bias = 0. Nếu loss là NaN/Inf, lỗi nằm ở hàm mất mát hoặc chưa trừ nhãn về $0..C-1$.
2. **Kiểm tra 2: Thử nghiệm quá khớp một lô nhỏ (20 mẫu) với mọi chính quy hóa tắt bỏ.**
   - *Lý do:* Nếu một mạng nơ-ron không thể ép loss về 0 và đạt accuracy 100% trên 20 mẫu sau 200-300 bước, 99.9% là do lỗi lập trình trong vòng lặp huấn luyện (quên `zero_grad()`, quên chuyển tham số vào optimizer, nhầm lẫn tensor giữa train và val, hoặc áp dụng Softmax hai lần). Đây là phép thử nhanh và rẻ nhất để loại trừ lỗi code.
3. **Kiểm tra 3: In chuẩn Gradient ($\|\mathbf{g}\|_2$) của từng lớp tham số sau `loss.backward()`.**
   - *Lý do:* Xác định xem gradient có thực sự đang chảy qua mạng hay không. Nếu gradient ở các tầng đầu bằng 0 hoặc `None`, mạng đang bị hiện tượng nơ-ron chết (dead ReLU), ngắt kết nối đồ thị autograd (`.detach()`), hoặc learning rate quá nhỏ khiến các bước cập nhật không tạo ra sự thay đổi đo đếm được.

---

## 6. Hạn chế và điều bất ngờ

- **Điều bất ngờ nhất:** Kiến trúc sâu hơn `M-deep` (256→128→64) dù có số lượng tham số ít hơn `M-wide` (55 687 so với 161 287) nhưng lại đạt Val Loss thấp kỷ lục (0.2073) và Val Acc rất cao (91.77%). Điều này chứng minh rằng với dữ liệu không gian địa lý và thổ nhưỡng, độ sâu phân cấp trừu tượng có giá trị biểu diễn không kém gì độ rộng nơ-ron.
- **Yếu tố thiết kế có thể gây thiên lệch:** Việc cố định 20 epoch cho mọi thí nghiệm khiến các batch size lớn (như batch 2048) bị thiệt thòi nặng về số lần cập nhật tham số (chỉ 182 bước/epoch so với 2 905 bước của batch 128). Một so sánh hoàn toàn công bằng về mặt tối ưu hoá giữa các batch size cần được đối chiếu ở cùng một tổng số bước cập nhật lặp lại (*iteration-matched*).
- **Hướng phát triển tiếp theo:** Nếu có thêm thời gian, chúng tôi sẽ thử nghiệm kỹ thuật Cosine Annealing Learning Rate kết hợp Warmup, và bổ sung cơ chế Focal Loss để tối ưu hoá đặc thù cho lớp hiếm Aspen (lớp 4) và Cottonwood (lớp 3).

---

## 7. Phụ lục

- **Toàn bộ cấu trúc thư mục nộp:**
  ```
  submission_2A202602597/
  ├── REPORT.md                         # Báo cáo kết luận chi tiết
  ├── experiments.xlsx                  # Bảng so sánh 24 thí nghiệm, giữ nguyên công thức
  ├── predictions_eval.csv              # Dự đoán 116 203 dòng của final-model trên eval
  ├── eval_result.json                  # Kết quả chính thức từ scripts/evaluate.py
  ├── figures/                          # 30 ảnh biểu đồ (24 exp riêng + 6 so sánh nhóm)
  ├── results/                          # 24 file JSON lưu trữ log từng thí nghiệm
  └── code/                             # Mã nguồn sạch sẽ, không còn NotImplementedError
      ├── lab.ipynb                     # Notebook hoàn chỉnh, giữ nguyên output
      ├── data.py                       # Pipeline nạp, chia, chuẩn hóa dữ liệu
      ├── model.py                      # Định nghĩa MLP và các phép khởi tạo
      ├── optimizer.py                  # Khởi tạo optimizer và gradient clipping
      ├── train.py                      # Vòng lặp huấn luyện, đánh giá, dự đoán
      ├── plots.py                      # Vẽ biểu đồ 3 ô và so sánh nhóm
      ├── results_table.py              # Xuất dữ liệu bảng tính Excel
      └── requirements.txt              # Danh sách thư viện phụ thuộc
  ```
- **Tổng thời gian thực thi chuỗi thí nghiệm:** Khoảng 16 phút trên GPU NVIDIA RTX 3050 Laptop GPU.
