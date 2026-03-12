# BÁO CÁO BÀI TẬP 3 - Lab 1
## Tiền Xử Lý Văn Bản cho Dịch Máy Nơ-ron (Neural Machine Translation)

**Sinh viên:** Egawa Masao  
**MSSV:** 3122411122  
**Lớp:** DCT122C5  
**Môn học:** Xử Lý Ngôn Ngữ Tự Nhiên (NLP)

---

## Mục tiêu

Bài tập lab 1 tập trung vào các bước tiền xử lý văn bản cơ bản cho hệ thống Dịch máy Nơ-ron (NMT), bao gồm: **tokenization** (tách từ), **xây dựng vocabulary** (từ điển), **encoding** (mã hóa câu), và hiểu khái niệm **BPE** (Byte Pair Encoding). Đây là những kỹ năng nền tảng bắt buộc trước khi xây dựng các mô hình Seq2Seq, Attention, và Transformer trong các Lab tiếp theo.

**Các công việc đã hoàn thành:**
- Cài đặt hàm `tokenize()` để tách từ cho song ngữ Anh-Việt
- Xây dựng lớp `Vocab` với special tokens (`<PAD>`, `<BOS>`, `<EOS>`)
- Mã hóa câu theo định dạng `[<BOS>, word_ids..., <EOS>]`
- Hiểu và giải thích phương pháp tokenization cấp subword (BPE)

---

## 1. GIỚI THIỆU

### 1.1. Bối cảnh

Dịch máy Nơ-ron (Neural Machine Translation - NMT) là một trong những ứng dụng quan trọng của Deep Learning trong xử lý ngôn ngữ tự nhiên. Khác với các phương pháp thống kê truyền thống, mạng nơ-ron hoạt động trên các biểu diễn số (numerical representations), do đó cần phải chuyển đổi văn bản thành dãy các số nguyên một cách có hệ thống.

### 1.2. Mục tiêu bài tập

1. **Tokenization:** Tách câu thành các token (từ)
2. **Vocabulary Construction:** Xây dựng từ điển với special tokens
3. **Sentence Encoding:** Mã hóa câu thành dãy số
4. **BPE Concept:** Hiểu khái niệm tokenization cấp subword

### 1.3. Dữ liệu

**Nguồn:** Corpus song ngữ Anh-Việt  
**Các file dữ liệu:**
- `data/train.en` - Câu tiếng Anh
- `data/train.vi` - Câu tiếng Việt
- **Tổng số cặp câu:** 10 cặp câu song ngữ

**Ví dụ một cặp câu:**
```
Tiếng Anh:  "I am a student"
Tiếng Việt: "tôi là sinh viên"
```

---

## 2. GIẢI PHÁP VÀ TRIỂN KHAI

### 2.1. Task 1: Tokenization (Tách từ)

#### 2.1.1. Yêu cầu

Cài đặt hàm `tokenize()` để chuyển đổi câu văn bản thành danh sách các từ (tokens).

**Ví dụ:**
```
Input:  "I am a student"
Output: ["i", "am", "a", "student"]
```

#### 2.1.2. Giải pháp

**Code triển khai trong `lab1_preprocessing.ipynb`:**

```python
def tokenize(sentence):
    """
    Tokenize một câu thành danh sách các từ
    - Chuyển về lowercase để chuẩn hóa
    - Tách bằng khoảng trắng
    """
    return sentence.lower().split()
```

**Giải thích thiết kế:**

1. **Lowercase normalization:** Chuyển tất cả về chữ thường giúp giảm kích thước vocabulary ("The" và "the" → cùng là "the")
2. **Whitespace splitting:** Tách theo khoảng trắng, phù hợp với tiếng Anh và tiếng Việt
3. **Đơn giản nhưng hiệu quả:** Phương pháp này đủ tốt cho bài lab giới thiệu

**Ưu điểm:**
- Dễ hiểu, dễ cài đặt
- Hoạt động tốt với corpus nhỏ
- Chuẩn hóa giúp giảm số lượng từ vựng

**Nhược điểm:**
- Không xử lý dấu câu riêng biệt
- Không xử lý các trường hợp đặc biệt (URLs, emails, số)

#### 2.1.3. Kết quả thực tế

**Test với câu tiếng Anh:**
```python
sentence = "I am a student"
tokens = tokenize(sentence)
# Result: ["i", "am", "a", "student"]
```

**Test với câu tiếng Việt:**
```python
sentence = "tôi là sinh viên"
tokens = tokenize(sentence)
# Result: ["tôi", "là", "sinh", "viên"]
```

 **Kết luận Task 1:** Hàm `tokenize()` hoạt động đúng.

---

### 2.2. Task 2: Vocabulary Construction (Xây dựng từ điển)

#### 2.2.1. Yêu cầu

Ta cần xây dựng từ điển ánh xạ giữa từ và ID số, **bắt buộc bao gồm special tokens:**

| Token | ID | Mục đích |
|-------|----|----|
| `<PAD>` | 0 | Padding (đệm) cho các câu ngắn trong batch |
| `<BOS>` | 1 | Begin-of-Sentence (bắt đầu câu) cho decoder |
| `<EOS>` | 2 | End-of-Sentence (kết thúc câu, điều kiện dừng) |

#### 2.2.2. Giải pháp

**Thiết kế lớp Vocab trong `lab1_preprocessing.ipynb`:**

```python
class Vocab:
    PAD_TOKEN = '<PAD>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'
    
    def __init__(self):
        """Khởi tạo từ điển rỗng và thêm special tokens"""
        self.word2id = {}  # Ánh xạ từ → ID
        self.id2word = {}  # Ánh xạ ID → từ
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Thêm special tokens với ID cố định"""
        self.word2id[self.PAD_TOKEN] = 0
        self.id2word[0] = self.PAD_TOKEN
        
        self.word2id[self.BOS_TOKEN] = 1
        self.id2word[1] = self.BOS_TOKEN
        
        self.word2id[self.EOS_TOKEN] = 2
        self.id2word[2] = self.EOS_TOKEN
    
    def add_word(self, w):
        """Thêm từ mới vào vocabulary"""
        if w not in self.word2id:
            idx = len(self.word2id)
            self.word2id[w] = idx
            self.id2word[idx] = w
    
    def build_vocab(self, sentences):
        """Xây dựng vocabulary từ danh sách câu"""
        for sentence in sentences:
            words = tokenize(sentence)
            for word in words:
                self.add_word(word)
    
    def encode(self, sentence, add_special_tokens=True):
        """Mã hóa câu thành dãy ID"""
        words = tokenize(sentence)
        ids = [self.word2id[word] for word in words if word in self.word2id]
        
        if add_special_tokens:
            ids = [self.word2id[self.BOS_TOKEN]] + ids + [self.word2id[self.EOS_TOKEN]]
        
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """Giải mã dãy ID về câu văn bản"""
        words = []
        for id in ids:
            if id in self.id2word:
                word = self.id2word[id]
                if skip_special_tokens and word in [self.PAD_TOKEN, 
                                                     self.BOS_TOKEN, 
                                                     self.EOS_TOKEN]:
                    continue
                words.append(word)
        return ' '.join(words)
    
    def __len__(self):
        """Trả về kích thước vocabulary"""
        return len(self.word2id)
```

**Các quyết định thiết kế quan trọng:**

1. **ID cố định cho special tokens (0, 1, 2):** Đảm bảo tính nhất quán giữa các lần chạy
2. **Ánh xạ hai chiều:** Cho phép cả encoding (mã hóa) và decoding (giải mã)
3. **Vocabulary động:** Các từ thông thường được gán ID từ 3 trở lên

#### 2.2.3. Kết quả thực tế

**Xây dựng vocabulary tiếng Anh:**

```python
vocab_en = Vocab()
vocab_en.build_vocab(en_sentences)

print(f"Tổng số tokens: {len(vocab_en)}")
# Output: Tổng số tokens: 33

print(f"Special tokens: 3")
print(f"Regular words: {len(vocab_en) - 3}")
# Output: Regular words: 30
```

**Ví dụ ánh xạ word2id:**
```
'<PAD>' -> ID: 0
'<BOS>' -> ID: 1
'<EOS>' -> ID: 2
'i'     -> ID: 3
'am'    -> ID: 4
'a'     -> ID: 5
'student' -> ID: 6
'she'   -> ID: 7
'is'    -> ID: 8
...
```

**Xây dựng vocabulary tiếng Việt:**

```python
vocab_vi = Vocab()
vocab_vi.build_vocab(vi_sentences)

print(f"Tổng số tokens: {len(vocab_vi)}")
# Output: Tổng số tokens: 37

print(f"Special tokens: 3")
print(f"Regular words: {len(vocab_vi) - 3}")
# Output: Regular words: 34
```

**Nhận xét:** Tiếng Việt có nhiều từ vựng hơn một chút (34 vs 30) do đặc điểm hình thái học và cú pháp (ví dụ: "một", "của", các từ ghép).

**Kết luận Task 2:** Lớp `Vocab` hoàn chỉnh với special tokens.

---

### 2.3. Task 3: Sentence Encoding (Mã hóa câu)

#### 2.3.1. Yêu cầu

Mã hóa câu văn bản thành dãy ID số theo định dạng:

**Format:** `[<BOS>, word_id_1, word_id_2, ..., <EOS>]`

**Ví dụ theo handout:**
```
Sentence: "I am a student"
Encoded:  [1, 3, 4, 5, 2]
```

Lưu ý: Ví dụ trong handout thiếu từ "a", trong implementation thực tế là:
```
Encoded: [1, 3, 4, 5, 6, 2]
         [<BOS>, i, am, a, student, <EOS>]
```

#### 2.3.2. Giải pháp

**Phương thức `encode()` đã triển khai trong lớp Vocab:**

```python
def encode(self, sentence, add_special_tokens=True):
    """
    Mã hóa câu thành dãy ID
    
    Parameters:
    - sentence: Câu văn bản cần mã hóa
    - add_special_tokens: Có thêm <BOS> và <EOS> hay không
    
    Returns:
    - List of integers (IDs)
    """
    words = tokenize(sentence)
    ids = [self.word2id[word] for word in words if word in self.word2id]
    
    if add_special_tokens:
        # Thêm <BOS> vào đầu, <EOS> vào cuối
        ids = [self.word2id[self.BOS_TOKEN]] + ids + [self.word2id[self.EOS_TOKEN]]
    
    return ids
```

**Phương thức `decode()` để kiểm tra:**

```python
def decode(self, ids, skip_special_tokens=True):
    """
    Giải mã dãy ID về câu văn bản
    
    Parameters:
    - ids: Danh sách các ID
    - skip_special_tokens: Bỏ qua special tokens khi giải mã
    
    Returns:
    - String (câu văn bản)
    """
    words = []
    for id in ids:
        if id in self.id2word:
            word = self.id2word[id]
            # Bỏ qua special tokens nếu cần
            if skip_special_tokens and word in [self.PAD_TOKEN, 
                                                 self.BOS_TOKEN, 
                                                 self.EOS_TOKEN]:
                continue
            words.append(word)
    return ' '.join(words)
```

#### 2.3.3. Kết quả thực tế

**Test case 1: Tiếng Anh**

```python
original = "I am a student"
encoded = vocab_en.encode(original, add_special_tokens=True)
decoded = vocab_en.decode(encoded, skip_special_tokens=True)

print(f"Original:  '{original}'")
print(f"Encoded:   {encoded}")
print(f"Decoded:   '{decoded}'")
print(f"Match:     {original.lower() == decoded}")
```

**Output:**
```
Original:  'I am a student'
Encoded:   [1, 3, 4, 5, 6, 2]
Chi tiết:  [<BOS>, i, am, a, student, <EOS>]
Decoded:   'i am a student'
Match:     True
```

**Test case 2: Tiếng Việt**

```python
original = "tôi là một sinh viên"
encoded = vocab_vi.encode(original, add_special_tokens=True)
decoded = vocab_vi.decode(encoded, skip_special_tokens=True)
```

**Output:**
```
Original:  'tôi là một sinh viên'
Encoded:   [1, 3, 4, 5, 6, 7, 2]
Chi tiết:  [<BOS>, tôi, là, một, sinh, viên, <EOS>]
Decoded:   'tôi là một sinh viên'
Match:     True
```

**Mã hóa toàn bộ dataset:**

```python
# Encode tất cả câu tiếng Anh
encoded_en_sentences = [vocab_en.encode(sent, add_special_tokens=True) 
                        for sent in en_sentences]

# Encode tất cả câu tiếng Việt
encoded_vi_sentences = [vocab_vi.encode(sent, add_special_tokens=True) 
                        for sent in vi_sentences]
```

✅ **Kết luận Task 3:** Mã hóa và giải mã hoạt động chính xác, định dạng `[<BOS>, words..., <EOS>]`.

---

### 2.4. Thống kê kết quả

#### 2.4.1. Thống kê Vocabulary

| Ngôn ngữ | Tổng số tokens | Special tokens | Từ vựng thực |
|----------|---------------|----------------|--------------|
| Tiếng Anh | 33 | 3 | 30 |
| Tiếng Việt | 37 | 3 | 34 |

#### 2.4.2. Thống kê độ dài câu (bao gồm `<BOS>` và `<EOS>`)

**Tiếng Anh:**
```
Độ dài trung bình: 6.2 tokens
Độ dài min:        5 tokens
Độ dài max:        9 tokens
```

**Tiếng Việt:**
```
Độ dài trung bình: 7.1 tokens
Độ dài min:        5 tokens
Độ dài max:        10 tokens
```

**Nhận xét:** Câu tiếng Việt có xu hướng dài hơn do cấu trúc cú pháp (sử dụng từ phân loại, từ ghép, v.v.).

#### 2.4.3. Ví dụ encode/decode hoàn chỉnh

**Cặp câu 1:**
```
EN text:    "I am a student"
EN encoded: [1, 3, 4, 5, 6, 2]
EN visual:  [<BOS>, i, am, a, student, <EOS>]

VI text:    "tôi là một sinh viên"
VI encoded: [1, 3, 4, 5, 6, 7, 2]
VI visual:  [<BOS>, tôi, là, một, sinh, viên, <EOS>]
```

**Cặp câu 2:**
```
EN text:    "She is my friend"
EN encoded: [1, 7, 8, 9, 10, 2]
EN visual:  [<BOS>, she, is, my, friend, <EOS>]

VI text:    "cô ấy là bạn của tôi"
VI encoded: [1, 8, 9, 4, 10, 11, 3, 2]
VI visual:  [<BOS>, cô, ấy, là, bạn, của, tôi, <EOS>]
```

---

## 3. KHÁI NIỆM BPE (Byte Pair Encoding)

### 3.1. Tại sao cần BPE?

**Hạn chế của tokenization cấp từ (word-level):**

1. **Vấn đề Out-of-Vocabulary (OOV):** Từ mới không có trong từ điển không thể mã hóa
2. **Vocabulary quá lớn:** Mỗi dạng từ là một token riêng biệt
3. **Không chia sẻ hình thái:** "run", "running", "runs" là 3 token hoàn toàn độc lập
4. **Xử lý kém từ hiếm:** Từ xuất hiện ít có biểu diễn kém

**Ví dụ vấn đề OOV:**
```
Training vocab: {"i", "am", "student"}
Test sentence: "I am a teacher"
→ Không thể encode "teacher" (chưa từng thấy)
```

### 3.2. Ý tưởng của BPE

**Khái niệm cốt lõi:** Tách từ thành các đơn vị subword (nhỏ hơn từ, lớn hơn ký tự) có thể tái sử dụng.

**Thuật toán BPE:**

1. **Khởi tạo:** Vocabulary = tất cả các ký tự trong corpus
2. **Lặp lại:**
   - Tìm cặp ký tự/subword xuất hiện nhiều nhất
   - Gộp (merge) cặp đó thành một token mới
   - Thêm token mới vào vocabulary
3. **Dừng lại:** Khi đạt kích thước vocabulary mong muốn (ví dụ: 30K tokens)

**Ví dụ minh họa:**

**Bước 1: Khởi tạo**
```
Corpus: ["low", "lower", "newest", "new"]
Initial vocab: {l, o, w, e, r, n, s, t}
Tokenized: l o w _, l o w e r _, n e w e s t _, n e w _
```

**Bước 2: Iteration 1**
```
Cặp xuất hiện nhiều nhất: "e" + "s" → "es" (2 lần)
Merge: n e w e s t → n e w es t
New vocab: {l, o, w, e, r, n, s, t, es}
```

**Bước 3: Iteration 2**
```
Cặp xuất hiện nhiều nhất: "es" + "t" → "est" (2 lần)
Merge: n e w es t → n e w est
New vocab: {l, o, w, e, r, n, s, t, es, est, ...}
```

**Kết quả sau nhiều iterations:**
```
Vocabulary cuối cùng:
{l, o, w, e, r, n, s, t, ..., "lo", "low", "new", "est", "newest"}

Tokenization examples:
"lowest"  → ["low", "est"]
"newer"   → ["new", "er"]
"unknown" → ["un", "know", "n"] hoặc ["u", "n", "k", "n", "o", "w", "n"]
```

### 3.3. So sánh Word-level vs BPE

| Tiêu chí | Word-level (Lab 1) | BPE (Subword) |
|----------|-------------------|---------------|
| **Kích thước vocab** | Nhỏ (30-40 với corpus nhỏ) | Trung bình (30K-50K) |
| **Xử lý OOV** |  Không encode được từ mới |  Tách thành subwords |
| **Hình thái học** |  Mỗi từ độc lập |  Chia sẻ subwords ("unhappy" = ["un", "happy"]) |
| **Từ hiếm** |  Biểu diễn kém |  Tách thành subwords đã biết |
| **Độ phức tạp** |  Đơn giản, dễ hiểu |  Phức tạp hơn, cần training |
| **Ứng dụng** | Giáo dục, corpus nhỏ | Hệ thống production, mô hình lớn |

### 3.4. BPE trong các hệ thống thực tế

**Các mô hình NLP hiện đại sử dụng BPE:**

- **GPT-3/GPT-4:** 50,257 BPE tokens
- **BERT:** WordPiece (biến thể của BPE), ~30K tokens
- **Google Translate:** Sử dụng BPE cho đa ngôn ngữ
- **Facebook FAIR:** SentencePiece + BPE

**Thư viện phổ biến:**
- `sentencepiece` (Google)
- `tokenizers` (HuggingFace)
- `subword-nmt` (Rico Sennrich)

### 3.5. Ví dụ tokenization so sánh

**Câu gốc:** "unhappiness"

**Word-level:**
```
Tokens: ["unhappiness"]
→ Nếu từ này không có trong training vocab → OOV error
```

**BPE (giả sử đã train):**
```
Tokens: ["un", "happiness"] hoặc ["un", "happy", "ness"]
→ Mỗi subword đều đã biết → không có OOV
→ "un" có thể tái sử dụng: "unhappy", "unknown", "unfair"
```

### 3.6. Khi nào dùng phương pháp nào?

**Dùng Word-level khi:**
- Corpus nhỏ, từ vựng hạn chế
- Mục đích giáo dục, học tập
- Cần đơn giản, dễ debug
- Không cần xử lý OOV

**Dùng BPE khi:**
- Corpus lớn, đa dạng
- Cần production-ready system
- Xử lý nhiều ngôn ngữ
- Cần robust với từ mới

---

## 4. TỔNG KẾT VÀ ĐÁNH GIÁ

### 4.1. Các công việc đã hoàn thành

 **Task 1: Tokenization**
- Cài đặt hàm `tokenize()` đơn giản và hiệu quả
- Xử lý tốt cả tiếng Anh và tiếng Việt
- Chuẩn hóa bằng lowercase

 **Task 2: Vocabulary Construction**
- Xây dựng lớp `Vocab` hoàn chỉnh
- Special tokens đúng ID: `<PAD>=0`, `<BOS>=1`, `<EOS>=2`
- Ánh xạ hai chiều: `word2id` và `id2word`
- Hỗ trợ thêm từ động

 **Task 3: Sentence Encoding**
- Mã hóa câu theo format `[<BOS>, words..., <EOS>]`
- Hàm `encode()` và `decode()` hoạt động chính xác
- Kiểm tra encoding/decoding: 100% match

### 4.2. Kết quả đạt được

**Dữ liệu đã chuẩn bị sẵn sàng cho các Lab tiếp theo:**

```
✓ vocab_en: Từ điển tiếng Anh (33 tokens)
✓ vocab_vi: Từ điển tiếng Việt (37 tokens)
✓ encoded_en_sentences: 10 câu tiếng Anh đã mã hóa
✓ encoded_vi_sentences: 10 câu tiếng Việt đã mã hóa
✓ Special tokens được cấu hình đúng
```

**Dữ liệu này sẽ được sử dụng trong:**
- **Lab 2:** Mô hình Seq2Seq với encoder-decoder (GRU/LSTM)
- **Lab 3:** Cơ chế Attention để cải thiện alignment
- **Lab 4:** Mô hình Transformer (state-of-the-art)

### 4.3. Kiến thức thu được

**Kỹ năng kỹ thuật:**
1. Tiền xử lý văn bản cho NLP
2. Quản lý vocabulary với special tokens
3. Mã hóa/giải mã sequence
4. Hiểu trade-offs giữa các phương pháp tokenization

**Hiểu biết khái niệm:**
1. Tại sao cần special tokens (`<PAD>`, `<BOS>`, `<EOS>`)
2. Cách tokenization ảnh hưởng đến hiệu năng mô hình
3. Sự phát triển từ word-level đến subword tokenization
4. Chuẩn bị cho các lab NMT tiếp theo
