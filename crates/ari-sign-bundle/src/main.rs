//! Standalone signing tool for Ari skill bundles.
//!
//! Three subcommands:
//!
//!   ari-sign-bundle gen-key <out-path>
//!     Generate a fresh Ed25519 keypair. The 32-byte private key is written
//!     to <out-path> with mode 0600. The hex-encoded public key is printed
//!     to stdout — pass this to `ari install --trust-key-hex` so the engine
//!     trusts bundles signed with this key.
//!
//!   ari-sign-bundle pubkey <key-path>
//!     Print the hex-encoded public key for an existing private key file.
//!
//!   ari-sign-bundle sign <bundle-path> <key-path>
//!     Compute SHA-256 of <bundle-path>, sign the digest with the private
//!     key from <key-path>, and write:
//!       <bundle-path>.sha256  — lowercase hex digest, no trailing newline
//!       <bundle-path>.sig     — raw 64-byte Ed25519 signature
//!
//! Random bytes for `gen-key` come from /dev/urandom directly. The engine
//! pulls in ed25519-dalek for verification anyway, so signing here doesn't
//! cost a new dependency.

use ed25519_dalek::{Signer, SigningKey, SECRET_KEY_LENGTH};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let cmd = match args.next() {
        Some(c) => c,
        None => {
            print_usage();
            return ExitCode::from(2);
        }
    };
    let result = match cmd.as_str() {
        "gen-key" => cmd_gen_key(args.collect()),
        "pubkey" => cmd_pubkey(args.collect()),
        "sign" => cmd_sign(args.collect()),
        "--help" | "-h" | "help" => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        other => Err(format!("unknown subcommand: {other}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ari-sign-bundle: {e}");
            ExitCode::from(1)
        }
    }
}

fn cmd_gen_key(args: Vec<String>) -> Result<(), String> {
    let out = args
        .first()
        .ok_or_else(|| "gen-key requires an output path".to_string())?;
    let out_path = PathBuf::from(out);

    if out_path.exists() {
        return Err(format!(
            "{} already exists — refusing to overwrite a key file",
            out_path.display()
        ));
    }

    let mut seed = [0u8; SECRET_KEY_LENGTH];
    read_random(&mut seed)?;
    let sk = SigningKey::from_bytes(&seed);
    let vk = sk.verifying_key();

    write_key_file(&out_path, &seed)?;
    println!("{}", hex_encode(vk.as_bytes()));
    eprintln!(
        "ari-sign-bundle: wrote private key to {} (mode 0600)",
        out_path.display()
    );
    eprintln!("ari-sign-bundle: pass the public key above to `ari install --trust-key-hex`");
    Ok(())
}

fn cmd_pubkey(args: Vec<String>) -> Result<(), String> {
    let key_path = args
        .first()
        .ok_or_else(|| "pubkey requires a key file path".to_string())?;
    let sk = read_signing_key(Path::new(key_path))?;
    println!("{}", hex_encode(sk.verifying_key().as_bytes()));
    Ok(())
}

fn cmd_sign(args: Vec<String>) -> Result<(), String> {
    let bundle = args
        .first()
        .ok_or_else(|| "sign requires <bundle> <key>".to_string())?;
    let key = args
        .get(1)
        .ok_or_else(|| "sign requires <bundle> <key>".to_string())?;
    let bundle_path = PathBuf::from(bundle);
    let key_path = PathBuf::from(key);

    let bytes = std::fs::read(&bundle_path)
        .map_err(|e| format!("could not read bundle {}: {e}", bundle_path.display()))?;
    let sk = read_signing_key(&key_path)?;

    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    let digest_hex = hex_encode(&digest);
    let sig = sk.sign(&digest).to_bytes();

    let sha_path = append_extension(&bundle_path, "sha256");
    let sig_path = append_extension(&bundle_path, "sig");
    std::fs::write(&sha_path, digest_hex.as_bytes())
        .map_err(|e| format!("could not write {}: {e}", sha_path.display()))?;
    std::fs::write(&sig_path, sig)
        .map_err(|e| format!("could not write {}: {e}", sig_path.display()))?;

    eprintln!("ari-sign-bundle: sha256 → {}", sha_path.display());
    eprintln!("ari-sign-bundle: signature → {}", sig_path.display());
    println!("{digest_hex}");
    Ok(())
}

fn read_signing_key(path: &Path) -> Result<SigningKey, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("could not read key file {}: {e}", path.display()))?;
    if bytes.len() != SECRET_KEY_LENGTH {
        return Err(format!(
            "key file {} is {} bytes, expected {}",
            path.display(),
            bytes.len(),
            SECRET_KEY_LENGTH
        ));
    }
    let mut seed = [0u8; SECRET_KEY_LENGTH];
    seed.copy_from_slice(&bytes);
    Ok(SigningKey::from_bytes(&seed))
}

fn write_key_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
        }
    }
    std::fs::write(path, bytes)
        .map_err(|e| format!("could not write key file {}: {e}", path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(path, perms)
            .map_err(|e| format!("could not chmod 0600 {}: {e}", path.display()))?;
    }
    Ok(())
}

fn read_random(buf: &mut [u8]) -> Result<(), String> {
    use std::io::Read;
    let mut f = std::fs::File::open("/dev/urandom")
        .map_err(|e| format!("could not open /dev/urandom: {e}"))?;
    f.read_exact(buf)
        .map_err(|e| format!("could not read /dev/urandom: {e}"))?;
    Ok(())
}

fn append_extension(path: &Path, ext: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

fn print_usage() {
    eprintln!("usage:");
    eprintln!("  ari-sign-bundle gen-key <out-path>");
    eprintln!("  ari-sign-bundle pubkey <key-path>");
    eprintln!("  ari-sign-bundle sign <bundle-path> <key-path>");
}
