# PDF to Markdown Conversion Workflow Guide

A complete guide for transferring PDF documents between your local personal computer and the remote VM instance, running OCR conversion with local **Qwen Coder** (or **Gemini** / **Gemma**), and downloading the resulting Obsidian Markdown and extracted images.

---

## 📋 Overview of the Pipeline

```
[Personal Computer]
        │
        │  1. scp (upload PDF)
        ▼
[VM: /home/jagadeesh28/Rust/pdf2md/input/]
        │
        │  2. cargo run --release -- (OCR & Markdown generation)
        ▼
[VM: /home/jagadeesh28/Rust/pdf2md/output/]
  ├── document.md
  └── images/
        │
        │  3. scp -r (download output folder)
        ▼
[Personal Computer: /path/to/local/Obsidian_Vault/]
```

---

## 🚀 Step 1: Upload PDF from Personal Computer to VM

Open terminal / PowerShell on your **local computer**:

### Upload Single File
```bash
scp /path/to/local/your_document.pdf jagadeesh28@<VM_IP_OR_HOSTNAME>:/home/jagadeesh28/Rust/pdf2md/input/
```

### Upload All PDFs in a Directory
```bash
 scp -r e:\Rust\pdf2md\input jagadeesh28@34.14.211.10:/home/jagadeesh28/Rust/pdf2md/
```

> [!TIP] Using SSH Keys
> If you authenticate using a private key (`.pem` or `.id_rsa`):
> ```bash
> scp -i ~/.ssh/your-key.pem /path/to/local/your_document.pdf jagadeesh28@<VM_IP_OR_HOSTNAME>:/home/jagadeesh28/Rust/pdf2md/input/
> ```

---

## ⚙️ Step 2: Run Extraction on the VM

### 1. Connect to the VM
```bash
ssh jagadeesh28@<VM_IP_OR_HOSTNAME>
```

### 2. Navigate to Project
```bash
cd /home/jagadeesh28/Rust/pdf2md
```

### 3. Ensure Local Model Service is Running (For Qwen Default)
If you run **Ollama** locally on the VM:
```bash
# Verify Ollama is running and has your preferred model:
ollama list
# e.g., pull if not already present:
ollama pull qwen2.5-coder:latest
```

*(Optional)* If your local model is served on a custom port or engine (vLLM / LM Studio / LocalAI):
```bash
export QWEN_URL="http://localhost:11434/v1"
export QWEN_MODEL="qwen2.5-coder:latest"
```

### 4. Execute `pdf2md`

```bash
# Syntax:
# cargo run --release -- <input.pdf> <output.md> <begin-page> <end-page> [backend]

# Default: Qwen Coder (Local)
cargo run --release -- input/your_document.pdf output/document.md 1 5

# Alternative: Google Gemini
export GEMINI_API_KEY="your_api_key_here"
cargo run --release -- input/your_document.pdf output/document.md 1 5 gemini

# Alternative: Google Gemma
export GEMMA_API_KEY="your_api_key_here"
cargo run --release -- input/your_document.pdf output/document.md 1 5 gemma
```

> [!NOTE] Output Directory Structure
> The conversion automatically creates:
> - `output/document.md`: Verbatim Markdown formatted with Obsidian LaTeX (`$...$` / `$$...$$`), headings, and tables.
> - `output/images/`: Extracted high-resolution PNG images (`page-1.png`, `page-2.png`, etc.).

---

## 📥 Step 3: Copy Output & Images Back to Personal Computer

Run this on your **local computer** terminal:

```bash
# Downloads the entire output directory into your target folder (e.g. your Obsidian vault)
scp -r jagadeesh28@<VM_IP_OR_HOSTNAME>:/home/jagadeesh28/Rust/pdf2md/output /path/to/local/Obsidian_Vault/
```

With SSH key:
```bash
scp -i ~/.ssh/your-key.pem -r jagadeesh28@<VM_IP_OR_HOSTNAME>:/home/jagadeesh28/Rust/pdf2md/output /path/to/local/Obsidian_Vault/
```

---

## 🗂️ Resulting Obsidian Vault Structure

Your local Obsidian vault will contain:
```text
Obsidian_Vault/
└── output/
    ├── document.md
    └── images/
        ├── page-1.png
        ├── page-2.png
        └── ...
```

In Obsidian:
- Mathematical equations render seamlessly using native MathJax/KaTeX.
- Tables, headers, and bullet structures are standard Obsidian Markdown.
- Embedded page image links (`![image-1](images/page-1.png)`) resolve automatically.

---

## 🛠️ Quick Reference Cheat Sheet

| Action               | Command                                                                 | Location  |
| :------------------- | :---------------------------------------------------------------------- | :-------- |
| **Upload PDF**       | `scp file.pdf jagadeesh28@<VM_IP>:/home/jagadeesh28/Rust/pdf2md/input/` | Local PC  |
| **Convert (Qwen)**   | `cargo run --release -- input/file.pdf output/file.md 1 10`             | Remote VM |
| **Convert (Gemini)** | `cargo run --release -- input/file.pdf output/file.md 1 10 gemini`      | Remote VM |
| **Convert (Gemma)**  | `cargo run --release -- input/file.pdf output/file.md 1 10 gemma`       | Remote VM |
| **Download Result**  | `scp -r jagadeesh28@<VM_IP>:/home/jagadeesh28/Rust/pdf2md/output ./`    | Local PC  |

without any connection to basic as daemon process
nohup cargo run --release -- /home/jagadeesh28/Rust/pdf2md/input/Five output > output/log.txt 2>&1 &
