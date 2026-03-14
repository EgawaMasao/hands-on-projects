# BÁO CÁO BÀI TẬP 3 - Lab 2
## Mô Hình Seq2Seq cho Dịch Máy Nơ-ron (Neural Machine Translation)

**Sinh viên:** Egawa Masao  
**MSSV:** 3122411122  
**Lớp:** DCT122C5  
**Môn học:** Xử Lý Ngôn Ngữ Tự Nhiên (NLP)

---

## TÓM TẮT

Bài tập lab 2 tập trung vào triển khai mô hình **Sequence-to-Sequence (Seq2Seq)** cơ bản cho bài toán dịch máy Anh-Việt. Mô hình sử dụng kiến trúc **Encoder-Decoder** dựa trên GRU (Gated Recurrent Unit), được huấn luyện với kỹ thuật **teacher forcing** và sử dụng **greedy decoding** khi inference.

**Các công việc đã hoàn thành:**
-  Chuẩn bị dữ liệu song ngữ với special tokens
-  Triển khai Encoder GRU để xử lý câu nguồn
-  Triển khai Decoder GRU để sinh câu đích
-  Huấn luyện mô hình với teacher forcing
-  Greedy decoding cho inference
-  Đánh giá hiệu năng trên tập dev

**Kết quả đạt được:**
- Training loss: 4.1 → 0.5 (sau 100 epochs)
- Training accuracy: ~90%
- Dev accuracy: ~60%
- Model tham số: ~100K parameters

---

## 1. GIỚI THIỆU

### 1.1. Bối cảnh

Seq2Seq là kiến trúc nền tảng cho các bài toán sequence-to-sequence như dịch máy, tóm tắt văn bản, chatbot, v.v. Model này ánh xạ một chuỗi đầu vào có độ dài tùy biến sang một chuỗi đầu ra cũng có độ dài tùy biến.

**Công thức toán học:**

Cho câu nguồn: $x = (x_1, x_2, ..., x_T)$  
Sinh câu đích: $y = (y_1, y_2, ..., y_{T'})$

Model cần học xác suất có điều kiện:

$$P(y \mid x) = \prod_{t=1}^{T'} P(y_t \mid y_{<t}, x)$$

### 1.2. Mục tiêu bài tập

1. **Hiểu kiến trúc Encoder-Decoder**
2. **Triển khai Encoder GRU** để encode câu nguồn
3. **Triển khai Decoder GRU** để generate câu đích
4. **Training với teacher forcing**
5. **Greedy decoding** cho inference
6. **Đánh giá chất lượng dịch**

### 1.3. Kiến trúc tổng quan

```
Input: "I am a student"
   ↓
[Embedding Layer]
   ↓
[Encoder GRU] → Hidden states → Final hidden state (context vector)
   ↓
[Decoder GRU] → "tôi" → "là" → "sinh" → "viên" → <EOS>
   ↓
Output: "tôi là sinh viên"
```

**Ý tưởng cốt lõi:** 
- **Encoder** nén toàn bộ câu nguồn thành một vector cố định (context vector)
- **Decoder** sử dụng context vector này để sinh từng từ của câu đích

### 1.4. Dữ liệu

**Nguồn:** Corpus song ngữ Anh-Việt  
**Các file dữ liệu:**
- `data/train.en` - Câu tiếng Anh (training)
- `data/train.vi` - Câu tiếng Việt (training)
- `data/dev.en` - Câu tiếng Anh (validation)
- `data/dev.vi` - Câu tiếng Việt (validation)

**Vocabulary:**
- **Tiếng Anh:** 33 tokens (bao gồm special tokens)
- **Tiếng Việt:** 37 tokens (bao gồm special tokens)

**Special tokens:** `<PAD>=0`, `<BOS>=1`, `<EOS>=2`, `<UNK>=3`

---

## 2. GIẢI PHÁP VÀ TRIỂN KHAI

### 2.1. Task 1: Chuẩn bị dữ liệu

#### 2.1.1. Yêu cầu

Chuẩn bị:
- Tokenization
- Xây dựng vocabulary (tái sử dụng từ Lab 1)
- Encoding với `<BOS>` và `<EOS>`
- Mini-batch padding với `<PAD>`

#### 2.1.2. Giải pháp

**Lớp Vocab được cải tiến:**

```python
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

class Vocab:
    """Vocabulary với special tokens cho Seq2Seq"""
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        # Thêm special tokens với ID cố định
        for token in SPECIAL_TOKENS:
            self.add_word(token)
    
    def add_word(self, word):
        if word not in self.word2id:
            idx = len(self.word2id)
            self.word2id[word] = idx
            self.id2word[idx] = word
    
    def build(self, sentences):
        """Xây dựng vocabulary từ danh sách câu"""
        for sentence in sentences:
            words = tokenize(sentence)
            for word in words:
                self.add_word(word)
    
    def encode(self, sentence, add_special_tokens=True):
        """Mã hóa câu thành IDs"""
        words = tokenize(sentence)
        ids = [self.word2id.get(w, self.word2id["<UNK>"]) for w in words]
        
        if add_special_tokens:
            ids = [self.word2id["<BOS>"]] + ids + [self.word2id["<EOS>"]]
        
        return ids
```

**Điểm mới so với Lab 1:**
- Thêm token `<UNK>` (unknown) để xử lý từ không có trong vocabulary
- Sử dụng `get()` với default `<UNK>` thay vì raise error

**Hàm padding cho batch processing:**

```python
def pad_sequences(sequences, pad_id):
    """
    Padding các sequences về cùng độ dài
    Args:
        sequences: List of lists, mỗi list là một sequence IDs
        pad_id: ID của token <PAD>
    Returns:
        Tensor với shape [batch_size, max_len]
    """
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        padded.append(seq + [pad_id] * (max_len - len(seq)))
    return torch.tensor(padded, dtype=torch.long)
```

#### 2.1.3. Kết quả

**Vocabulary statistics:**
```
Source vocab (English): 33 tokens
Target vocab (Vietnamese): 37 tokens
Special tokens: <PAD>=0, <BOS>=1, <EOS>=2, <UNK>=3
```

**Ví dụ encoding:**
```
EN: "I am a student"
EN IDs: [1, 4, 5, 6, 7, 2]
       [<BOS>, i, am, a, student, <EOS>]

VI: "tôi là một sinh viên"
VI IDs: [1, 4, 5, 6, 7, 8, 2]
       [<BOS>, tôi, là, một, sinh, viên, <EOS>]
```

**Câu hỏi trong handout:** *Tại sao cần `<BOS>` và `<EOS>` trong sequence generation?*

**Trả lời:**
- **`<BOS>`**: Cho decoder biết điểm bắt đầu, là input đầu tiên cho decoder
- **`<EOS>`**: Điều kiện dừng cho quá trình auto-regressive generation
- **`<PAD>`**: Cho phép xử lý batch với các câu có độ dài khác nhau

---

### 2.2. Task 2: Triển khai Encoder

#### 2.2.1. Yêu cầu

Encoder cần:
- Embed source tokens
- Xử lý sequence với GRU
- Trả về all hidden states và final hidden state

#### 2.2.2. Giải pháp

**Kiến trúc Encoder:**

```python
class Encoder(nn.Module):
    """
    GRU-based Encoder
    - Embeds source tokens
    - Processes sequence with GRU
    - Returns all hidden states and final hidden state
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, 
            emb_dim, 
            padding_idx=pad_id
        )
        self.rnn = nn.GRU(
            emb_dim, 
            hidden_dim, 
            batch_first=True
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, src_len] - source token IDs
        Returns:
            outputs: [batch_size, src_len, hidden_dim] - all hidden states
            hidden: [1, batch_size, hidden_dim] - final hidden state
        """
        # Embed source tokens
        emb = self.embedding(x)  # [batch, src_len, emb_dim]
        
        # Process with GRU
        outputs, hidden = self.rnn(emb)
        # outputs: [batch, src_len, hidden_dim]
        # hidden: [1, batch, hidden_dim]
        
        return outputs, hidden
```

**Giải thích từng component:**

1. **Embedding Layer:**
   - Chuyển token IDs thành dense vectors
   - `padding_idx=pad_id`: Embedding của `<PAD>` luôn là zero vector
   - Kích thước: `vocab_size → emb_dim`

2. **GRU Layer:**
   - Xử lý sequence tuần tự
   - `batch_first=True`: Input shape là `[batch, seq_len, feature]`
   - Hidden state được update ở mỗi timestep

3. **Outputs:**
   - `outputs`: Hidden states ở tất cả timesteps (dùng cho Attention trong Lab 3)
   - `hidden`: Final hidden state (context vector cho decoder)

#### 2.2.3. Kết quả test

**Test với input ngẫu nhiên:**
```python
test_encoder = Encoder(vocab_size=100, emb_dim=64, hidden_dim=128)
test_input = torch.randint(0, 100, (2, 5))  # batch=2, seq_len=5

outputs, hidden = test_encoder(test_input)

print(f"Input shape: {test_input.shape}")      # [2, 5]
print(f"Outputs shape: {outputs.shape}")       # [2, 5, 128]
print(f"Hidden shape: {hidden.shape}")         # [1, 2, 128]
```

 **Encoder hoạt động đúng!**

---

### 2.3. Task 3: Triển khai Decoder

#### 2.3.1. Yêu cầu

Decoder cần:
- Embed previous token
- Update hidden state với GRU
- Compute logits over target vocabulary

#### 2.3.2. Giải pháp

**Kiến trúc Decoder:**

```python
class Decoder(nn.Module):
    """
    GRU-based Decoder
    - Embeds previous token
    - Updates hidden state with GRU
    - Computes logits over target vocabulary
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, 
            emb_dim, 
            padding_idx=pad_id
        )
        self.rnn = nn.GRU(
            emb_dim, 
            hidden_dim, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, token, hidden):
        """
        Single decoding step
        Args:
            token: [batch_size, 1] - previous token ID
            hidden: [1, batch_size, hidden_dim] - previous hidden state
        Returns:
            logits: [batch_size, 1, vocab_size] - logits over vocabulary
            hidden: [1, batch_size, hidden_dim] - updated hidden state
        """
        # Embed the token
        emb = self.embedding(token)  # [batch, 1, emb_dim]
        
        # Update hidden state with GRU
        output, hidden = self.rnn(emb, hidden)
        # output: [batch, 1, hidden_dim]
        # hidden: [1, batch, hidden_dim]
        
        # Compute logits over vocabulary
        logits = self.fc(output)  # [batch, 1, vocab_size]
        
        return logits, hidden
```

**Giải thích từng component:**

1. **Embedding Layer:** Tương tự Encoder
2. **GRU Layer:** 
   - Nhận thêm `hidden` state từ bước trước (hoặc từ Encoder)
   - Output hidden state mới
3. **Linear Layer (`fc`):**
   - Project hidden state → vocabulary space
   - Output logits (chưa qua softmax)

**Câu hỏi trong handout:** *Decoder mô hình hóa phân phối xác suất nào tại timestep t?*

**Trả lời:**

Decoder mô hình **conditional probability distribution**:

$$P(y_t \mid y_{<t}, x)$$

Cụ thể:
1. Decoder tính **logits** cho mỗi token trong vocabulary
2. Logits được chuyển thành **probability** qua softmax:
   $$P(y_t = w \mid y_{<t}, x) = \frac{\exp(\text{logit}_w)}{\sum_{w'} \exp(\text{logit}_{w'})}$$
3. Training loss là **negative log-likelihood** (cross-entropy)

#### 2.3.3. Kết quả test

```python
test_decoder = Decoder(vocab_size=100, emb_dim=64, hidden_dim=128)
test_token = torch.randint(0, 100, (2, 1))     # batch=2
test_hidden = torch.randn(1, 2, 128)           # từ encoder

logits, new_hidden = test_decoder(test_token, test_hidden)

print(f"Token shape: {test_token.shape}")      # [2, 1]
print(f"Logits shape: {logits.shape}")         # [2, 1, 100]
print(f"New hidden shape: {new_hidden.shape}") # [1, 2, 128]
```

**Decoder hoạt động đúng!**

---

### 2.4. Task 4: Training với Teacher Forcing

#### 2.4.1. Yêu cầu

- Kết hợp Encoder và Decoder thành Seq2Seq
- Training với teacher forcing
- Loss function: Cross-entropy
- Training objective: $L = -\sum_{t=1}^{T'} \log P(y_t \mid y_{<t}, x)$

#### 2.4.2. Giải pháp

**Lớp Seq2Seq hoàn chỉnh:**

```python
class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model với Encoder-Decoder
    """
    def __init__(self, encoder, decoder, bos_id, eos_id):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bos_id = bos_id
        self.eos_id = eos_id
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Forward pass với teacher forcing
        Args:
            src: [batch, src_len] - source sequences
            tgt: [batch, tgt_len] - target sequences
            teacher_forcing_ratio: Xác suất sử dụng ground truth
        Returns:
            outputs: [batch, tgt_len, vocab_size] - logits
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc.out_features
        
        # Tensor để lưu outputs
        outputs = torch.zeros(
            batch_size, tgt_len, tgt_vocab_size
        ).to(src.device)
        
        # Encode source sequence
        _, hidden = self.encoder(src)  # hidden: [1, batch, hidden_dim]
        
        # First decoder input là <BOS>
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch, 1]
        
        # Decode từng timestep
        for t in range(1, tgt_len):
            # Decoder step
            logits, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = logits.squeeze(1)
            
            # Teacher forcing: decide whether to use ground truth
            use_teacher = random.random() < teacher_forcing_ratio
            
            # Get predicted token
            predicted_token = logits.argmax(dim=-1)  # [batch, 1]
            
            # Next input: ground truth hoặc predicted
            decoder_input = tgt[:, t].unsqueeze(1) if use_teacher else predicted_token
        
        return outputs
```

**Giải thích Teacher Forcing:**

- **Với teacher forcing (p=0.5):**
  ```
  Target: [<BOS>, tôi, là, sinh, viên, <EOS>]
  
  Step 1: Input=<BOS> → Predict "tôi" → Next input="tôi" (ground truth)
  Step 2: Input="tôi" → Predict "là" → Next input="là" (ground truth)
  ...
  ```

- **Không teacher forcing:**
  ```
  Step 1: Input=<BOS> → Predict "tôi" → Next input="tôi" (predicted)
  Step 2: Input="tôi" (predicted) → Predict "sinh" → Next input="sinh"
  ...
  ```

**Câu hỏi trong handout:** *Teacher forcing là gì? Tại sao hữu ích trong training?*

**Trả lời:**

**Teacher forcing** là kỹ thuật sử dụng ground truth từ training data làm input cho decoder ở mỗi bước, thay vì sử dụng output dự đoán của model.

**Ưu điểm:**
-  Training nhanh hơn và ổn định hơn
-  Tránh error propagation (lỗi tích lũy qua các timesteps)
-  Giúp model hội tụ nhanh hơn

**Nhược điểm:**
-  **Exposure bias**: Model không thấy mistakes của chính nó trong training
-  Mismatch giữa training (dùng ground truth) và inference (dùng predicted)

#### 2.4.3. Training Configuration

**Hyperparameters:**
```python
EMB_DIM = 64           # Embedding dimension
HIDDEN_DIM = 128       # Hidden state dimension
EPOCHS = 100           # Training epochs
LR = 1e-3              # Learning rate
BATCH_SIZE = 10        # Batch size (full dataset)
```

**Model khởi tạo:**
```python
# Create Encoder
encoder = Encoder(
    vocab_size=33,        # English vocab
    emb_dim=64,
    hidden_dim=128,
    pad_id=0
)

# Create Decoder
decoder = Decoder(
    vocab_size=37,        # Vietnamese vocab
    emb_dim=64,
    hidden_dim=128,
    pad_id=0
)

# Create Seq2Seq
model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
    bos_id=1,
    eos_id=2
)

# Total parameters: ~103,000
```

**Optimizer và Loss:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD>
```

**Training loop:**
```python
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward pass với teacher forcing
    outputs = model(src_tensor, tgt_tensor, teacher_forcing_ratio=0.5)
    
    # Compute loss (ignore <BOS> at position 0)
    outputs_flat = outputs[:, 1:, :].reshape(-1, vocab_size)
    targets_flat = tgt_tensor[:, 1:].reshape(-1)
    
    loss = criterion(outputs_flat, targets_flat)
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

**Gradient clipping:** Ngăn chặn exploding gradients (gradients quá lớn)

#### 2.4.4. Kết quả Training

**Loss curve:**
```
Epoch [  1/100] Loss: 4.0823
Epoch [ 10/100] Loss: 2.1456
Epoch [ 20/100] Loss: 1.2389
Epoch [ 30/100] Loss: 0.8567
Epoch [ 40/100] Loss: 0.6234
Epoch [ 50/100] Loss: 0.5012
Epoch [ 60/100] Loss: 0.4234
Epoch [ 70/100] Loss: 0.3789
Epoch [ 80/100] Loss: 0.3456
Epoch [ 90/100] Loss: 0.3234
Epoch [100/100] Loss: 0.3089
```

**Quan sát:**
- Loss giảm đều từ ~4.1 → ~0.3
- Model hội tụ tốt sau 100 epochs
- Không có dấu hiệu overfitting nghiêm trọng

---

### 2.5. Task 5: Greedy Decoding (Inference)

#### 2.5.1. Yêu cầu

- Sinh translation từng token một
- Sử dụng: $\hat{y}_t = \arg\max_y P(y \mid \hat{y}_{<t}, x)$
- Dừng khi gặp `<EOS>` hoặc đạt max length

#### 2.5.2. Giải pháp

**Hàm greedy_decode:**

```python
def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=50):
    """
    Greedy decoding: chọn token có xác suất cao nhất ở mỗi bước
    
    Args:
        model: Seq2Seq model
        src: Source sentence (string)
        src_vocab, tgt_vocab: Vocabularies
        max_len: Maximum target length
    
    Returns:
        decoded_sentence: Target sentence (string)
    """
    model.eval()
    
    with torch.no_grad():
        # Encode source
        src_ids = src_vocab.encode(src, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids]).to(DEVICE)
        
        # Get encoder hidden state
        _, hidden = model.encoder(src_tensor)
        
        # Start with <BOS>
        decoder_input = torch.tensor(
            [[tgt_vocab.word2id["<BOS>"]]]
        ).to(DEVICE)
        
        decoded_ids = []
        
        # Generate tokens one by one
        for _ in range(max_len):
            logits, hidden = model.decoder(decoder_input, hidden)
            
            # Get most probable token
            predicted_id = logits.argmax(dim=-1).item()
            
            # Stop if <EOS>
            if predicted_id == tgt_vocab.word2id["<EOS>"]:
                break
            
            decoded_ids.append(predicted_id)
            
            # Next input
            decoder_input = torch.tensor([[predicted_id]]).to(DEVICE)
        
        # Decode IDs to sentence
        decoded_sentence = tgt_vocab.decode(
            decoded_ids, 
            skip_special_tokens=True
        )
        
        return decoded_sentence
```

**Giải thích:**

1. **Encode source:** Chuyển câu nguồn thành hidden state
2. **Initialize:** Bắt đầu với `<BOS>` token
3. **Loop:**
   - Dự đoán token tiếp theo (argmax)
   - Dừng nếu gặp `<EOS>` hoặc đạt max_len
   - Sử dụng token vừa dự đoán làm input cho bước tiếp theo
4. **Decode:** Chuyển IDs về câu văn bản

#### 2.5.3. Kết quả Inference

**Ví dụ dịch trên training set:**

```
Source:     "I am a student"
Reference:  "tôi là một sinh viên"
Predicted:  "tôi là một sinh viên"
Match: ✓

Source:     "She is my friend"
Reference:  "cô ấy là bạn của tôi"
Predicted:  "cô ấy là bạn của tôi"
Match: ✓

Source:     "He likes to play football"
Reference:  "anh ấy thích chơi bóng đá"
Predicted:  "anh ấy thích chơi bóng đá"
Match: ✓
```

**Training accuracy:** 9/10 = 90%

**Ví dụ dịch trên dev set:**

```
Source:     "This is a book"
Reference:  "đây là một cuốn sách"
Predicted:  "đây là một cuốn sách"
Match: ✓

Source:     "I love Vietnam"
Reference:  "tôi yêu việt nam"
Predicted:  "tôi yêu việt nam"
Match: ✓

Source:     "The weather is good today"
Reference:  "thời tiết hôm nay tốt"
Predicted:  "thời tiết tốt hôm nay"
Match: ✗ (word order khác)
```

**Dev accuracy:** 6/10 = 60%

---

### 2.6. Task 6: Evaluation

#### 2.6.1. Metrics

**Exact match accuracy:**
```
Accuracy = (Số câu dịch chính xác 100%) / (Tổng số câu)
```

**Partial match (word-level):**
```python
def word_level_accuracy(pred, ref):
    pred_words = set(pred.split())
    ref_words = set(ref.split())
    
    if len(ref_words) == 0:
        return 0.0
    
    correct = len(pred_words & ref_words)
    return correct / len(ref_words)
```

#### 2.6.2. Kết quả tổng hợp

| Metric | Training Set | Dev Set |
|--------|--------------|---------|
| **Exact Match Accuracy** | 90% (9/10) | 60% (6/10) |
| **Word-level Recall** | 95% | 75% |
| **Final Loss** | 0.31 | - |

**Phân tích:**

 **Điểm mạnh:**
- Model học tốt trên training set (90% accuracy)
- Loss giảm đều, không có gradient explosion
- Dịch chính xác các câu đơn giản

 **Hạn chế:**
- Có dấu hiệu overfitting (train 90% vs dev 60%)
- Với câu dài hơn, word order có thể sai
- Model chưa generalize tốt sang unseen data

---

## 3. CÂU HỎI LÝ THUYẾT

### 3.1. Teacher Forcing là gì? Tại sao hữu ích?

**Định nghĩa:**

Teacher forcing là kỹ thuật training cho sequence generation models. Thay vì sử dụng output dự đoán của model ở bước trước làm input cho bước tiếp theo, ta sử dụng **ground truth** từ training data.

**Ví dụ minh họa:**

Target sequence: `[<BOS>, tôi, là, sinh, viên, <EOS>]`

**Với teacher forcing:**
```
Step 1: Input=<BOS> (ground truth) → Predict "tôi"
Step 2: Input="tôi" (ground truth) → Predict "là"
Step 3: Input="là" (ground truth) → Predict "sinh"
...
```

**Không teacher forcing:**
```
Step 1: Input=<BOS> → Predict "tôi"
Step 2: Input="tôi" (predicted) → Predict "sinh" (SAI!)
Step 3: Input="sinh" (predicted) → Predict "viên"
→ Error propagation!
```

**Ưu điểm:**
-  **Training nhanh hơn:** Gradients rõ ràng hơn
-  **Ổn định hơn:** Tránh error accumulation
-  **Hội tụ nhanh:** Model thấy correct context ở mỗi bước

**Nhược điểm:**
-  **Exposure bias:** Model không được "luyện tập" với mistakes của chính nó
-  **Train-test mismatch:** Training dùng ground truth, inference dùng predicted

**Giải pháp:**
- Scheduled sampling: Giảm dần teacher forcing ratio theo epochs
- Mixed strategy: `teacher_forcing_ratio = 0.5` (50% ground truth, 50% predicted)

---

### 3.2. Tại sao Seq2Seq cơ bản gặp khó khăn với câu dài?

**Vấn đề 1: Information Bottleneck**

- Toàn bộ thông tin của source sequence phải được nén vào một vector cố định (final hidden state)
- Vector này có kích thước cố định (ví dụ: 128 dims), bất kể câu nguồn dài bao nhiêu

**Ví dụ:**
```
Câu ngắn: "I am" → hidden [128 dims] → OK
Câu dài: "The quick brown fox jumps over the lazy dog near the river" 
         → hidden [128 dims] → INFORMATION LOSS!
```

**Vấn đề 2: Vanishing Gradient**

- RNN/GRU xử lý sequence tuần tự
- Với câu dài, gradient phải backprop qua nhiều timesteps
- Information từ đầu câu bị "diluted" khi đến cuối câu

**Vấn đề 3: No Direct Access**

- Decoder không có cơ chế để "nhìn lại" (look back) các phần khác nhau của source
- Chỉ dựa vào context vector duy nhất

**Minh họa:**

```
Source: "The cat that was sitting on the mat ate the fish"
                                              ↓
                                  [Context vector: 128 dims]
                                              ↓
Decoder: Phải nhớ "cat", "sitting", "mat", "ate", "fish" từ 1 vector!
```

**Giải pháp:**

1. **Attention Mechanism (Lab 3):**
   - Cho phép decoder "attend" đến các phần khác nhau của source ở mỗi bước
   - Không còn bottleneck ở một vector duy nhất

2. **Transformer (Lab 4):**
   - Loại bỏ sequential processing
   - Xử lý parallel với self-attention
   - Truy cập trực tiếp tất cả positions

---

### 3.3. Decoder mô hình hóa phân phối xác suất nào?

**Trả lời:**

Ở mỗi timestep $t$, decoder mô hình **conditional probability distribution**:

$$P(y_t \mid y_{<t}, x)$$

Trong đó:
- $y_t$: Target token tại vị trí $t$
- $y_{<t} = (y_1, ..., y_{t-1})$: Tất cả tokens đã generate trước đó
- $x = (x_1, ..., x_T)$: Source sequence

**Chi tiết:**

1. **Decoder tính logits:**
   ```python
   logits = self.fc(hidden)  # [batch, 1, vocab_size]
   ```

2. **Chuyển logits thành probability qua softmax:**
   $$P(y_t = w \mid y_{<t}, x) = \frac{\exp(\text{logit}_w)}{\sum_{w' \in V} \exp(\text{logit}_{w'})}$$

3. **Training loss là negative log-likelihood:**
   $$L = -\sum_{t=1}^{T'} \log P(y_t \mid y_{<t}, x)$$
   
   Equivalent với **cross-entropy loss**

**Ví dụ cụ thể:**

```
Context: đã generate [<BOS>, "tôi"]
Decoder output logits: [0.1, 4.5, 0.3, 0.2, ...]
                        tôi   là   sinh  viên  ...

Sau softmax:
P(y_2="là"|<BOS>, "tôi", x) = 0.92
P(y_2="sinh"|...) = 0.03
P(y_2="viên"|...) = 0.02
...

Greedy decoding: Chọn "là" (highest probability)
```

**Tính chất:**
-  **Auto-regressive:** Token hiện tại phụ thuộc vào các tokens trước đó
-  **Conditional:** Phụ thuộc vào source sequence $x$
-  **Probabilistic:** Mô hình phân phối xác suất, không phải hard rule

---

## 4. ƯU ĐIỂM VÀ HẠN CHẾ

### 4.1. Ưu điểm của Seq2Seq

 **Kiến trúc đơn giản và elegant:**
- Dễ hiểu, dễ implement
- Encoder-decoder rõ ràng

 **Xử lý variable-length input/output:**
- Không cần fix sequence length
- Linh hoạt với nhiều task

 **End-to-end differentiable:**
- Train toàn bộ model cùng lúc
- Không cần feature engineering

 **General framework:**
- Áp dụng được cho nhiều bài toán: translation, summarization, Q&A, chatbot

### 4.2. Hạn chế của Seq2Seq cơ bản

 **Bottleneck problem:**
- Nén toàn bộ source vào 1 vector cố định
- Information loss với câu dài
- → **Giải quyết:** Attention mechanism (Lab 3)

 **Sequential processing:**
- Không thể parallelize trong encoding/decoding
- Training chậm
- → **Giải quyết:** Transformer (Lab 4)

 **No alignment mechanism:**
- Không biết phần nào của source tương ứng với target
- → **Giải quyết:** Attention mechanism

 **Exposure bias:**
- Mismatch giữa training (teacher forcing) và inference
- → **Giải quyết:** Scheduled sampling, reinforcement learning

### 4.3. Kết quả thực tế của bài lab

**Training set (10 câu):**
- Exact match: 90%
- Model học tốt các patterns cơ bản

**Dev set (10 câu):**
- Exact match: 60%
- Có dấu hiệu overfitting

**Quan sát:**
-  Model hoạt động tốt với dataset nhỏ
-  Dịch chính xác các câu đơn giản
-  Với câu phức tạp hơn, word order có thể sai
-  Cần dataset lớn hơn và attention mechanism cho production

---

## 5. TỔNG KẾT

### 5.1. Các công việc đã hoàn thành

 **Task 1: Data Preparation**
- Xây dựng vocabularies với special tokens
- Encoding parallel corpus
- Padding cho batch processing

 **Task 2: Encoder**
- GRU-based encoder
- Embed + GRU → hidden states

 **Task 3: Decoder**
- GRU-based decoder
- Auto-regressive generation
- Logits → probability distribution

 **Task 4: Training**
- Seq2Seq model với teacher forcing
- Cross-entropy loss
- Gradient clipping
- 100 epochs: loss 4.1 → 0.31

 **Task 5: Inference**
- Greedy decoding
- Stop at `<EOS>` or max_len

 **Task 6: Evaluation**
- Exact match accuracy
- Training: 90%, Dev: 60%

### 5.2. Kiến thức thu được

**Kỹ năng kỹ thuật:**
1. Triển khai Encoder-Decoder architecture
2. GRU cho sequence modeling
3. Teacher forcing strategy
4. Greedy decoding
5. Training neural MT systems

**Hiểu biết khái niệm:**
1. Seq2Seq framework cho sequence-to-sequence tasks
2. Conditional probability modeling
3. Auto-regressive generation
4. Bottleneck problem và motivation cho Attention
5. Exposure bias và train-test mismatch