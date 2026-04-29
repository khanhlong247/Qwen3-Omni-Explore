# TRIỂN KHAI VÀ THỬ NGHIỆM TOOL-CALLING ĐA PHƯƠNG THỨC VỚI QWEN2.5-OMNI

## 1. Giới thiệu mô hình Qwen2.5-Omni

Qwen2.5-Omni là một mô hình ngôn ngữ đa phương thức (Multimodal Large Language Model), được thiết kế để xử lý và hiểu đồng thời nhiều loại dữ liệu đầu vào bao gồm văn bản, hình ảnh, video và âm thanh. Khác với các hệ thống truyền thống phải qua bước chuyển đổi Audio-to-Text (ASR), Qwen2.5-Omni có khả năng nghe trực tiếp các đặc trưng âm thanh, từ đó phản hồi bằng văn bản hoặc tiếng nói một cách tự nhiên và liền mạch.

Hệ thống được xây dựng trên nền tảng kiến trúc Transformer, tối ưu hóa cho các tác vụ Agentic Workflow, cho phép nó không chỉ trò chuyện mà còn có thể gọi các hàm (Function Calling) để tương tác với hệ thống bên ngoài.

## 2. Cơ chế Thinker-Talker

Điểm mấu chốt giúp Qwen2.5-Omni xử lý được nhiều loại dữ liệu đầu vào trực tiếp nằm ở khả năng đưa mọi loại dữ liệu về một không gian vector thống nhất (Unified Vector Space) trước khi đi vào suy luận.

- Đồng nhất hóa đặc trưng (Feature Alignment): Thay vì chuyển âm thanh thành văn bản (ASR), mô hình sử dụng các bộ mã hóa chuyên biệt như Audio Encoder để trích xuất đặc trưng log-mel từ sóng âm.

    - Các đặc trưng này sau đó đi qua các lớp Linear Projection để ánh xạ (map) từ không gian vector của âm thanh và hình ảnh khớp hoàn toàn với không gian vector ẩn (hidden dimension) của các token văn bản.

- Thinker (Bộ suy luận thống nhất): Sau khi được ánh xạ, các đặc trưng âm thanh được chèn trực tiếp vào chuỗi nhúng văn bản (inputs_embeds) thông qua cơ chế masked_scatter.

    - Lúc này, đối với Thinker, một đoạn âm thanh hay một từ ngữ đều chỉ là một chuỗi các vector liên tục. Điều này cho phép cơ chế Self-Attention tính toán mối tương quan trực tiếp giữa "tiếng nói" và "văn bản" mà không cần bất kỳ bước dịch trung gian nào, giúp bảo toàn nguyên vẹn các thông tin về cảm xúc, âm điệu và ngữ cảnh.

- Talker (Bộ điều phối giọng nói): Tiếp nhận trạng thái ẩn (hidden states) từ Thinker sau khi đã xử lý xong context đa phương thức để dự đoán các speech tokens. Nó đóng vai trò cầu nối giữa suy luận logic và khả năng biểu đạt âm thanh.

- Token2Wav (Bộ chuyển đổi âm thanh): Sử dụng kiến trúc DiT (Diffusion Transformer) kết hợp với bộ giải mã BigVGAN để chuyển đổi các speech tokens thành sóng âm thanh (waveform) thực tế.

Cơ chế này cho phép mô hình có luồng suy nghĩ (Thought process) riêng trên một dòng dữ liệu hỗn hợp, giúp Agent đưa ra quyết định hành động hoặc gọi Tool Call với độ trễ thấp và độ chính xác cao.

## 3. Data Flow

Quy trình xử lý dữ liệu trong hệ thống thử nghiệm được thiết lập như sau:

- **Tiền xử lý đa phương thức (Preprocessing):** Các file âm thanh (.wav, .mp3) được trích xuất đặc trưng thông qua `process_mm_info` và chuyển đổi thành dạng Numpy array với tần số lấy mẫu 24,000Hz.

- **Mã hóa Tensor (Processor):** Qwen2_5OmniProcessor nhận văn bản prompt và đặc trưng âm thanh để tạo ra `input_features` và `input_ids` tương ứng, nạp vào bộ nhớ GPU.

- **Suy luận lần 1 (Thinker Inference):** Mô hình nạp Base Model kết hợp với LoRA Adapter từ checkpoint đã finetune. Thinker phân tích âm thanh để trích xuất thông tin (ví dụ: tên khách hàng, số hợp đồng) và sinh ra đoạn mã JSON gọi Tool.

- **Thực thi Tool (Execution):** Hệ thống bắt các tag `<tool_call>`, giải mã tham số và thực thi hàm Python tương ứng.

- **Suy luận lần 2 (Final Response):** Kết quả từ Tool (Observation) được nạp ngược lại vào hội thoại để Thinker đưa ra câu trả lời cuối cùng bằng ngôn ngữ tự nhiên.

## 4. Thiết lập thử nghiệm Audio Tool-calling

Thử nghiệm được thực hiện trên kịch bản Trợ lý Bảo hiểm ảo, sử dụng các công cụ cụ thể:

- Danh sách hàm: `verify_policy` (xác thực hợp đồng), `process_medical_claim` (bồi thường y tế), `process_accident_claim` (bồi thường tai nạn).

- Dữ liệu đầu vào: Ghép nối nhiều file audio của người dùng chứa các thông tin cá nhân như họ tên và mã số bảo hiểm.

- Prompting: Sử dụng `SYSTEM_INSTRUCTION` nghiêm ngặt để ép mô hình phải phân tích âm thanh và gọi tool ngay lập tức.

- Cấu hình phần cứng: Chạy trên môi trường Kaggle với GPU NVIDIA Tesla T4.

## 5. Kết quả thử nghiệm

Qua quá trình chạy thực tế, các kết quả chính được ghi nhận:

- Khả năng trích xuất: Mô hình đạt độ chính xác cao khi trích xuất thông tin định danh trực tiếp từ giọng nói (Ví dụ: policy_id: "HD113734").

- Tốc độ xử lý: Việc sử dụng định dạng 4-bit (NF4) giúp tiết kiệm VRAM đáng kể và đảm bảo thời gian suy luận (Inference) mượt mà trên phần cứng giới hạn.

- Thách thức: Trong các lượt chạy đầu, mô hình đôi khi gặp lỗi định dạng JSON (dùng nháy đơn thay vì nháy kép) hoặc xuất hiện hiện tượng ảo giác(hallucination) khi tự bịa ra thông tin bệnh án nếu không được nhắc nhở chặt chẽ bằng Prompt.

- Giải pháp: Áp dụng bộ parser linh hoạt (Universal Parser) và chèn tin nhắn hỗ trợ (hint) ở lượt suy luận cuối đã giúp Agent phản hồi tiếng Việt chuẩn xác và chuyên nghiệp.

## 6. Kết luận

Qwen2.5-Omni thể hiện tiềm năng vượt trội trong việc xây dựng các Agent hỗ trợ khách hàng thế hệ mới, nơi âm thanh được xử lý như một loại dữ liệu nguyên bản. Với việc nạp thành công các Adapter finetune, mô hình đã chứng minh được khả năng học sâu các nghiệp vụ bảo hiểm chuyên biệt và thực thi logic gọi hàm từ audio một cách tin cậy. Để triển khai thương mại, cần chú trọng thêm vào bước kiểm soát định dạng đầu ra (Format Validation) và tối ưu hóa luồng hội thoại để giảm thiểu tối đa hiện tượng ảo giác thông tin.

## 7. Data và Source code thử nghiệm

- Python notebook: [Qwen2.5-Omni Audio Function Call Version 3](https://www.kaggle.com/code/khanhlong/qwen2-5-omni-audio-function-call-version3)

- Base model: [Qwen2.5-Omni Base Model](https://www.kaggle.com/models/khanhlong/qwen2-5-omni)

- Finetuned model: [Qwen2.5-Omni Finetuned model](https://www.kaggle.com/models/khanhlong/qwen2-5-finetuned)

- Test Data: [Audio testing data](https://www.kaggle.com/datasets/khanhlong/conversation-test)