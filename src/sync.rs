use anyhow::{Context, Result, anyhow};
use std::env;
use std::path::{Path, PathBuf};

/// Configuration for syncing extracted Markdown and figures over an SSH connection (e.g. reverse tunnel).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub user: Option<String>,
    pub remote_dir: Option<PathBuf>,
    pub key_path: Option<PathBuf>,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: "127.0.0.1".to_string(),
            port: 2222,
            user: None,
            remote_dir: None,
            key_path: None,
        }
    }
}

impl SyncConfig {
    /// Loads sync configuration from environment variables:
    /// - `PDF2MD_SYNC`: "1", "true", "yes" to enable
    /// - `PDF2MD_SYNC_HOST`: remote host (default: "127.0.0.1")
    /// - `PDF2MD_SYNC_PORT`: SSH port (default: 2222)
    /// - `PDF2MD_SYNC_USER`: SSH username
    /// - `PDF2MD_SYNC_DIR`: Destination directory on remote host
    /// - `PDF2MD_SYNC_KEY`: Path to private SSH key
    pub fn from_env() -> Self {
        let enabled = env::var("PDF2MD_SYNC")
            .map(|v| {
                let v = v.trim().to_lowercase();
                v == "1" || v == "true" || v == "yes"
            })
            .unwrap_or(false);

        let host = env::var("PDF2MD_SYNC_HOST")
            .unwrap_or_else(|_| "127.0.0.1".to_string());

        let port = env::var("PDF2MD_SYNC_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(2222);

        let user = env::var("PDF2MD_SYNC_USER").ok().filter(|s| !s.is_empty());
        let remote_dir = env::var("PDF2MD_SYNC_DIR").ok().filter(|s| !s.is_empty()).map(PathBuf::from);
        let key_path = env::var("PDF2MD_SYNC_KEY").ok().filter(|s| !s.is_empty()).map(PathBuf::from);

        Self {
            enabled,
            host,
            port,
            user,
            remote_dir,
            key_path,
        }
    }
}

/// Parses command-line arguments related to SSH sync, returning `(consumed_count, env_key, env_val)`.
pub fn parse_sync_arg(arg: &str, next_arg: Option<&str>) -> Option<(usize, &'static str, String)> {
    if arg == "--sync" {
        return Some((1, "PDF2MD_SYNC", "1".to_string()));
    }
    if arg == "--sync-host" {
        return next_arg.map(|v| (2, "PDF2MD_SYNC_HOST", v.to_string()));
    }
    if let Some(val) = arg.strip_prefix("--sync-host=") {
        return Some((1, "PDF2MD_SYNC_HOST", val.to_string()));
    }
    if arg == "--sync-port" {
        return next_arg.map(|v| (2, "PDF2MD_SYNC_PORT", v.to_string()));
    }
    if let Some(val) = arg.strip_prefix("--sync-port=") {
        return Some((1, "PDF2MD_SYNC_PORT", val.to_string()));
    }
    if arg == "--sync-user" {
        return next_arg.map(|v| (2, "PDF2MD_SYNC_USER", v.to_string()));
    }
    if let Some(val) = arg.strip_prefix("--sync-user=") {
        return Some((1, "PDF2MD_SYNC_USER", val.to_string()));
    }
    if arg == "--sync-dir" {
        return next_arg.map(|v| (2, "PDF2MD_SYNC_DIR", v.to_string()));
    }
    if let Some(val) = arg.strip_prefix("--sync-dir=") {
        return Some((1, "PDF2MD_SYNC_DIR", val.to_string()));
    }
    if arg == "--sync-key" {
        return next_arg.map(|v| (2, "PDF2MD_SYNC_KEY", v.to_string()));
    }
    if let Some(val) = arg.strip_prefix("--sync-key=") {
        return Some((1, "PDF2MD_SYNC_KEY", val.to_string()));
    }
    None
}

/// Invokes SSH sync if enabled in `config`.
///
/// NOTE: This function is strictly non-fatal. If the SSH connection fails or
/// drops mid-transfer, it prints a clean warning and returns without failing
/// the underlying conversion or deleting local files.
pub fn sync_if_enabled(md_path: &Path, images_dir: &Path, config: &SyncConfig) {
    if !config.enabled {
        return;
    }

    let target_dest = match &config.user {
        Some(u) => format!("{}@{}:{}", u, config.host, config.port),
        None => format!("{}:{}", config.host, config.port),
    };

    println!("\n[Sync] 🚀 Initiating file sync over SSH tunnel to {target_dest}...");
    match sync_files(md_path, images_dir, config) {
        Ok(count) => {
            println!("[Sync] ✓ Successfully synced {count} file(s) to remote machine.");
        }
        Err(err) => {
            eprintln!(
                "[Sync] ⚠️  Warning: File transfer over SSH tunnel failed: {err:#}\n\
                 [Sync] All generated files remain safely preserved on this machine:\n\
                 [Sync]   Markdown: {}\n\
                 [Sync]   Images:   {}",
                md_path.display(),
                images_dir.display(),
            );
        }
    }
}

#[cfg(unix)]
fn sync_files(md_path: &Path, images_dir: &Path, config: &SyncConfig) -> Result<usize> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to initialize Tokio runtime for SSH sync")?;

    rt.block_on(sync_files_async(md_path, images_dir, config))
}

#[cfg(unix)]
async fn sync_files_async(md_path: &Path, images_dir: &Path, config: &SyncConfig) -> Result<usize> {
    use openssh::{KnownHosts, SessionBuilder};
    use openssh_sftp_client::Sftp;

    let mut builder = SessionBuilder::default();
    builder.port(config.port);
    builder.known_hosts_check(KnownHosts::Accept);
    builder.connect_timeout(std::time::Duration::from_secs(15));
    builder.server_alive_interval(std::time::Duration::from_secs(10));

    if let Some(ref user) = config.user {
        builder.user(user.clone());
    }
    if let Some(ref key) = config.key_path {
        builder.keyfile(key);
    }

    let destination = match &config.user {
        Some(u) => format!("ssh://{u}@{}", config.host),
        None => format!("ssh://{}", config.host),
    };

    let session = builder
        .connect(&destination)
        .await
        .with_context(|| format!("failed to establish SSH connection to {destination}"))?;

    let sftp = Sftp::from_session(session, openssh_sftp_client::SftpOptions::default())
        .await
        .context("failed to initialize SFTP subsystem")?;

    let mut fs = sftp.fs();
    let mut total_synced = 0;

    let base_remote = config
        .remote_dir
        .as_ref()
        .map(|d| d.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();

    if !base_remote.is_empty() {
        ensure_remote_dir(&mut fs, &base_remote).await?;
    }

    // 1. Sync markdown file
    if md_path.is_file() {
        let md_filename = md_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("output.md");
        let remote_md_path = if base_remote.is_empty() {
            md_filename.to_string()
        } else {
            format!("{}/{}", base_remote.trim_end_matches('/'), md_filename)
        };

        let bytes = upload_file(&sftp, md_path, &remote_md_path).await?;
        println!(
            "  -> Synced '{}' ({} bytes)",
            md_path.display(),
            bytes
        );
        total_synced += 1;
    }

    // 2. Sync images directory
    if images_dir.is_dir() {
        let folder_name = images_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("images");
        let remote_images_dir = if base_remote.is_empty() {
            folder_name.to_string()
        } else {
            format!("{}/{}", base_remote.trim_end_matches('/'), folder_name)
        };

        ensure_remote_dir(&mut fs, &remote_images_dir).await?;

        let mut entries = tokio::fs::read_dir(images_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            if entry_path.is_file() {
                let filename = entry.file_name();
                let filename_str = filename.to_string_lossy();
                let remote_file = format!("{remote_images_dir}/{filename_str}");
                let bytes = upload_file(&sftp, &entry_path, &remote_file).await?;
                println!(
                    "  -> Synced figure '{}' ({} bytes)",
                    filename_str,
                    bytes
                );
                total_synced += 1;
            }
        }
    }

    sftp.close()
        .await
        .context("failed to cleanly close SFTP session")?;

    Ok(total_synced)
}

#[cfg(unix)]
async fn ensure_remote_dir(fs: &mut openssh_sftp_client::fs::Fs, dir_path: &str) -> Result<()> {
    let clean = dir_path.replace('\\', "/");
    let parts: Vec<&str> = clean.split('/').filter(|p| !p.is_empty()).collect();

    let is_absolute = clean.starts_with('/');
    let mut current = if is_absolute { String::from("/") } else { String::new() };

    for part in parts {
        if !current.is_empty() && !current.ends_with('/') {
            current.push('/');
        }
        current.push_str(part);

        // Skip Windows-style drive specs like "C:" if encountered
        if current.len() == 2 && current.ends_with(':') {
            continue;
        }

        let _ = fs.create_dir(Path::new(&current)).await;
    }
    Ok(())
}

#[cfg(unix)]
async fn upload_file(
    sftp: &openssh_sftp_client::Sftp,
    local_path: &Path,
    remote_path_str: &str,
) -> Result<u64> {
    use tokio::io::AsyncWriteExt;

    let mut local_file = tokio::fs::File::open(local_path)
        .await
        .with_context(|| format!("failed to open local file {}", local_path.display()))?;

    let remote_file = sftp
        .create(Path::new(remote_path_str))
        .await
        .with_context(|| format!("failed to open remote file {remote_path_str}"))?;

    let mut compat_remote = openssh_sftp_client::file::TokioCompatFile::new(remote_file);

    let bytes = tokio::io::copy(&mut local_file, &mut compat_remote)
        .await
        .with_context(|| format!("failed to stream data to remote file {remote_path_str}"))?;

    compat_remote
        .flush()
        .await
        .with_context(|| format!("failed to flush remote file {remote_path_str}"))?;

    Ok(bytes)
}

#[cfg(not(unix))]
fn sync_files(md_path: &Path, images_dir: &Path, config: &SyncConfig) -> Result<usize> {
    use std::process::Command;

    let mut total_synced = 0;
    let port_str = config.port.to_string();

    let remote_target = match &config.user {
        Some(u) => format!("{}@{}:", u, config.host),
        None => format!("{}:", config.host),
    };

    let base_remote = config
        .remote_dir
        .as_ref()
        .map(|d| d.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();

    // 1. Sync markdown file
    if md_path.is_file() {
        let dest = format!("{remote_target}{base_remote}");
        let mut cmd = Command::new("scp");
        cmd.args(["-P", &port_str, "-o", "StrictHostKeyChecking=no"]);
        if let Some(ref key) = config.key_path {
            cmd.arg("-i").arg(key);
        }
        cmd.arg(md_path).arg(&dest);

        let status = cmd.status().context("failed to execute scp command")?;
        if !status.success() {
            return Err(anyhow!("scp exited with non-zero status: {status}"));
        }
        println!("  -> Synced '{}'", md_path.display());
        total_synced += 1;
    }

    // 2. Sync images directory
    if images_dir.is_dir() {
        let dest = format!("{remote_target}{base_remote}");
        let mut cmd = Command::new("scp");
        cmd.args(["-r", "-P", &port_str, "-o", "StrictHostKeyChecking=no"]);
        if let Some(ref key) = config.key_path {
            cmd.arg("-i").arg(key);
        }
        cmd.arg(images_dir).arg(&dest);

        let status = cmd.status().context("failed to execute scp command")?;
        if !status.success() {
            return Err(anyhow!("scp directory sync failed with status: {status}"));
        }
        println!("  -> Synced images directory '{}'", images_dir.display());
        total_synced += 1;
    }

    Ok(total_synced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_config_default() {
        let config = SyncConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 2222);
        assert_eq!(config.user, None);
        assert_eq!(config.remote_dir, None);
    }

    #[test]
    fn test_sync_config_from_env() {
        unsafe {
            env::set_var("PDF2MD_SYNC", "1");
            env::set_var("PDF2MD_SYNC_HOST", "127.0.0.1");
            env::set_var("PDF2MD_SYNC_PORT", "2222");
            env::set_var("PDF2MD_SYNC_USER", "testuser");
            env::set_var("PDF2MD_SYNC_DIR", "/home/testuser/transfers");
        }

        let config = SyncConfig::from_env();
        assert!(config.enabled);
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 2222);
        assert_eq!(config.user.as_deref(), Some("testuser"));
        assert_eq!(
            config.remote_dir.as_deref(),
            Some(Path::new("/home/testuser/transfers"))
        );

        unsafe {
            env::remove_var("PDF2MD_SYNC");
            env::remove_var("PDF2MD_SYNC_HOST");
            env::remove_var("PDF2MD_SYNC_PORT");
            env::remove_var("PDF2MD_SYNC_USER");
            env::remove_var("PDF2MD_SYNC_DIR");
        }
    }

    #[test]
    fn test_parse_sync_arg() {
        assert_eq!(
            parse_sync_arg("--sync", None),
            Some((1, "PDF2MD_SYNC", "1".to_string()))
        );
        assert_eq!(
            parse_sync_arg("--sync-port", Some("3333")),
            Some((2, "PDF2MD_SYNC_PORT", "3333".to_string()))
        );
        assert_eq!(
            parse_sync_arg("--sync-user=alice", None),
            Some((1, "PDF2MD_SYNC_USER", "alice".to_string()))
        );
        assert_eq!(
            parse_sync_arg("--sync-dir=/dest/path", None),
            Some((1, "PDF2MD_SYNC_DIR", "/dest/path".to_string()))
        );
        assert_eq!(
            parse_sync_arg("--sync-key", Some("~/.ssh/id_ed25519")),
            Some((2, "PDF2MD_SYNC_KEY", "~/.ssh/id_ed25519".to_string()))
        );
        assert_eq!(parse_sync_arg("--other-flag", None), None);
    }
}
