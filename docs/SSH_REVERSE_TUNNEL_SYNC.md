# Remote Sync Guide: SSH Reverse Tunnel & OpenSSH SFTP

This guide explains how to automatically sync converted Markdown files (`<pdfname>.md`) and extracted figures (`<pdfname>_images/`) from a remote cloud VM instance (GCP / AWS) back to your local laptop in real time, preventing data loss from network drops, SSH timeouts, or VM preemption.

---

## 1. Architecture Overview

```
+------------------------------------+             +-----------------------------------------+
|        Local Laptop                |             |          Cloud VM Instance              |
|                                    |             |                                         |
|  1. Local OpenSSH Server running   |  Reverse    |  3. Runs `pdf2md`                       |
|     on port 22                     |  Tunnel     |  4. Writes files locally to disk first  |
|                                    |             |  5. `openssh` crate connects to         |
|  2. Opens reverse SSH tunnel:      |<------------|     `127.0.0.1:2222`                    |
|     `ssh -R 2222:localhost:22 ...` |             |  6. Streams files via SFTP to laptop    |
+------------------------------------+             +-----------------------------------------+
```

### Key Safety Guarantees
- **Local-First Preservation**: All Markdown and figure files are saved locally on the VM's disk **before** any network transfer begins.
- **Non-Fatal Synchronization**: If the tunnel drops or the network fails, `pdf2md` catches the error, prints a clean warning, and continues. The conversion **never fails or aborts**, and no files are lost.
- **Immediate Batch Delivery**: In batch mode (`pdf2md <input-dir> <output-dir>`), each PDF's `.md` file and its `<pdfname>_images/` folder are synced immediately upon completion of that PDF, so you don't have to wait for the entire batch to finish.

---

## 2. One-Time Setup on Local Laptop

Ensure your local laptop can accept incoming SSH connections over `localhost:22`:

### On Windows Laptop:
1. Open PowerShell as Administrator and check OpenSSH Server status:
   ```powershell
   Get-Service -Name sshd
   ```
2. If stopped or not installed:
   ```powershell
   Start-Service sshd
   Set-Service -Name sshd -StartupType 'Automatic'
   ```
3. Test local login:
   ```powershell
   ssh localhost
   ```

### On macOS / Linux Laptop:
- **macOS**: System Settings -> General -> Sharing -> Enable **Remote Login**.
- **Linux**: `sudo systemctl enable --now ssh`

---

## 3. Step-by-Step Execution Workflow

### Step 1: Open the Reverse Tunnel from Your Laptop
Open a terminal on your laptop and run:
```bash
ssh -R 2222:localhost:22 <vm-user>@<vm-ip-or-dns>
```
> [!IMPORTANT]
> **Keep this terminal window open** while conversions are running. The reverse tunnel remains active as long as this SSH session is alive.

*(Optional)* If your local SSH server is passwordless with an SSH key, ensure your public key (`~/.ssh/id_ed25519.pub` or `id_rsa.pub`) is added to your laptop's authorized keys so `openssh` connects without prompting for a password.

---

### Step 2: Run `pdf2md` on the VM Instance

#### Option A: Using CLI Flags
```bash
# Single PDF conversion
pdf2md input.pdf output/ \
  --sync \
  --sync-user <your-laptop-username> \
  --sync-dir /path/on/laptop/transfers

# Batch directory conversion
pdf2md input_dir/ output_dir/ \
  --sync \
  --sync-user <your-laptop-username> \
  --sync-dir /path/on/laptop/transfers
```

#### Option B: Using Environment Variables (Recommended for Shell Profiles)
Export the variables once in your terminal session or add them to `~/.bashrc`:
```bash
export PDF2MD_SYNC=1
export PDF2MD_SYNC_USER=your_laptop_username
export PDF2MD_SYNC_DIR=/Users/username/Desktop/Transfers    # or E:/Rust/Transfers on Windows

# Now run pdf2md normally — sync happens automatically!
pdf2md input.pdf output/
```

---

## 4. Configuration Reference

| CLI Option | Environment Variable | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `--sync` | `PDF2MD_SYNC` | *(disabled)* | Set to `1`, `true`, or `yes` to enable auto-sync |
| `--sync-host <host>` | `PDF2MD_SYNC_HOST` | `127.0.0.1` | Remote host address (localhost for reverse tunnel) |
| `--sync-port <port>` | `PDF2MD_SYNC_PORT` | `2222` | Forwarded reverse tunnel port |
| `--sync-user <user>` | `PDF2MD_SYNC_USER` | *(current user)* | SSH username on your local laptop |
| `--sync-dir <dir>` | `PDF2MD_SYNC_DIR` | *(remote landing dir)*| Target directory on your laptop |
| `--sync-key <path>` | `PDF2MD_SYNC_KEY` | *(default SSH keys)* | Optional path to private SSH key file on VM |

---

## 5. What Gets Transferred

For each converted PDF:
1. **Markdown Document**: `<pdfname>.md`
2. **Figures Directory**: `<pdfname>_images/` containing all extracted, enhanced figures and diagrams referenced by the Markdown file (`figure_p1_1.png`, etc.).

---

## 6. Troubleshooting & Tips

### Tunnel Disconnection / Network Drop
If your laptop goes to sleep or Wi-Fi drops:
- `pdf2md` will print:
  ```
  [Sync] ⚠️  Warning: File transfer over SSH tunnel failed: ...
  [Sync] All generated files remain safely preserved on this machine:
  [Sync]   Markdown: output/document.md
  [Sync]   Images:   output/document_images
  ```
- The PDF conversion proceeds normally without interruption. Once reconnected, you can re-run or scp the remaining files from the VM output directory.

### Preventing SSH Tunnel Timeouts
To keep your laptop's reverse tunnel connection alive through NAT routers and cloud firewalls, add this to your laptop's `~/.ssh/config`:
```ssh-config
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 3
```
Or specify it directly in the command:
```bash
ssh -o ServerAliveInterval=60 -R 2222:localhost:22 <vm-user>@<vm-ip>
```
