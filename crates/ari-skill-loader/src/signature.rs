//! Ed25519 signature verification for skill bundles.
//!
//! The engine **only verifies**; it never signs. Signing happens in the
//! registry's CI workflow with a key that lives as a GitHub Actions secret
//! and is never accessible to clients. The corresponding public key is baked
//! into the engine binary at build time as a `[u8; 32]` constant.
//!
//! ## Wire format
//!
//! What we sign is **the SHA-256 of the bundle tarball**, not the tarball
//! bytes directly. This is standard practice (apt, dnf, Homebrew, ...): the
//! signature is small and constant-time, the hash is whatever the bundle
//! produces, and verification only needs to redo the hash and check the
//! signature against it.
//!
//! Signature files on disk are raw 64-byte Ed25519 signatures. No PGP
//! armoring, no PEM, no envelopes. Equally minimal: the verifying key is
//! 32 raw bytes.
//!
//! ## Compromise model
//!
//! If the registry signing key is compromised, the recovery path is:
//!
//! 1. Generate a new keypair locally
//! 2. Replace the GitHub Actions secret
//! 3. Re-sign every existing bundle with the new key
//! 4. Cut a new engine release with **both** the new and old pubkeys, with
//!    the new one as primary. The engine accepts either during the rotation
//!    window so users on the old engine can still install bundles signed
//!    with the new key once they update.
//! 5. After one engine release cycle, drop the old pubkey entirely.
//!
//! See `GOVERNANCE.md` in the registry repo for the full procedure.

use ed25519_dalek::{Signature, Verifier, VerifyingKey, SIGNATURE_LENGTH};
use thiserror::Error;

/// Length of an Ed25519 verifying key in bytes.
pub const VERIFYING_KEY_LENGTH: usize = 32;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SignatureError {
    #[error("verifying key must be {VERIFYING_KEY_LENGTH} bytes, got {0}")]
    BadKeyLength(usize),

    #[error("verifying key bytes are not a valid Ed25519 point")]
    BadKey,

    #[error("signature must be {SIGNATURE_LENGTH} bytes, got {0}")]
    BadSignatureLength(usize),

    #[error("signature did not verify against the trusted key")]
    Invalid,
}

/// A trust root: one or more verifying keys, any of which can validate a
/// signature. The single-key case is the common one; the multi-key case
/// covers the rotation window when both old and new keys are accepted.
#[derive(Debug, Clone)]
pub struct TrustRoot {
    keys: Vec<VerifyingKey>,
}

impl TrustRoot {
    /// Build a trust root from one verifying key (the common case).
    pub fn single(key_bytes: &[u8]) -> Result<Self, SignatureError> {
        let key = parse_verifying_key(key_bytes)?;
        Ok(Self { keys: vec![key] })
    }

    /// Build a trust root from multiple verifying keys. Used during a key
    /// rotation window where the engine accepts signatures from either the
    /// new or old key.
    pub fn multi(key_bytes_list: &[&[u8]]) -> Result<Self, SignatureError> {
        let keys = key_bytes_list
            .iter()
            .map(|b| parse_verifying_key(b))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { keys })
    }

    /// Verify `signature` against the SHA-256 hash bytes `digest` using any
    /// of the trusted keys. Returns Ok(()) on the first key that accepts the
    /// signature, or `SignatureError::Invalid` if none of them do.
    pub fn verify(&self, digest: &[u8], signature_bytes: &[u8]) -> Result<(), SignatureError> {
        if signature_bytes.len() != SIGNATURE_LENGTH {
            return Err(SignatureError::BadSignatureLength(signature_bytes.len()));
        }
        // `Signature::from_bytes` takes a fixed-size array reference. After
        // the length check above, the slice is exactly the right size, so
        // the conversion is infallible.
        let array: &[u8; SIGNATURE_LENGTH] = signature_bytes
            .try_into()
            .expect("length checked above");
        let sig = Signature::from_bytes(array);
        for key in &self.keys {
            if key.verify(digest, &sig).is_ok() {
                return Ok(());
            }
        }
        Err(SignatureError::Invalid)
    }

    /// Number of trusted keys. Useful for "we're in a rotation window" log
    /// messages on engine startup.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

fn parse_verifying_key(bytes: &[u8]) -> Result<VerifyingKey, SignatureError> {
    if bytes.len() != VERIFYING_KEY_LENGTH {
        return Err(SignatureError::BadKeyLength(bytes.len()));
    }
    let array: &[u8; VERIFYING_KEY_LENGTH] = bytes.try_into().expect("length checked above");
    VerifyingKey::from_bytes(array).map_err(|_| SignatureError::BadKey)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Signer;
    use ed25519_dalek::SigningKey;

    /// Build a fresh keypair from a deterministic seed so test failures
    /// reproduce. Test-only — production keys come from a real CSPRNG.
    fn fixed_keypair(seed: u8) -> (SigningKey, VerifyingKey) {
        let bytes = [seed; 32];
        let sk = SigningKey::from_bytes(&bytes);
        let vk = sk.verifying_key();
        (sk, vk)
    }

    #[test]
    fn verifying_key_length_constant_matches_dalek() {
        assert_eq!(VERIFYING_KEY_LENGTH, 32);
    }

    #[test]
    fn round_trip_single_key() {
        let (sk, vk) = fixed_keypair(7);
        let trust = TrustRoot::single(vk.as_bytes()).unwrap();
        let digest = b"some sha256 digest bytes!!!!!!!!"; // 32 bytes
        let sig = sk.sign(digest);
        assert!(trust.verify(digest, &sig.to_bytes()).is_ok());
    }

    #[test]
    fn signature_from_a_different_key_is_rejected() {
        let (_, real_vk) = fixed_keypair(1);
        let (impostor_sk, _) = fixed_keypair(2);
        let trust = TrustRoot::single(real_vk.as_bytes()).unwrap();
        let digest = b"the same digest the real signer would have sig"; // doesn't matter
        let bad_sig = impostor_sk.sign(digest);
        assert_eq!(
            trust.verify(digest, &bad_sig.to_bytes()).unwrap_err(),
            SignatureError::Invalid
        );
    }

    #[test]
    fn tampered_digest_is_rejected() {
        let (sk, vk) = fixed_keypair(3);
        let trust = TrustRoot::single(vk.as_bytes()).unwrap();
        let digest = [0xAAu8; 32];
        let sig = sk.sign(&digest);
        let mut tampered = digest;
        tampered[0] ^= 1;
        assert_eq!(
            trust.verify(&tampered, &sig.to_bytes()).unwrap_err(),
            SignatureError::Invalid
        );
    }

    #[test]
    fn bad_key_length_rejected() {
        let err = TrustRoot::single(&[0u8; 31]).unwrap_err();
        assert_eq!(err, SignatureError::BadKeyLength(31));
    }

    #[test]
    fn bad_signature_length_rejected() {
        let (_, vk) = fixed_keypair(4);
        let trust = TrustRoot::single(vk.as_bytes()).unwrap();
        let err = trust.verify(b"digest", &[0u8; 63]).unwrap_err();
        assert_eq!(err, SignatureError::BadSignatureLength(63));
    }

    #[test]
    fn rotation_window_accepts_signatures_from_either_key() {
        // Simulates the engine release that has both the new and the old
        // pubkey baked in.
        let (old_sk, old_vk) = fixed_keypair(5);
        let (new_sk, new_vk) = fixed_keypair(6);
        let trust = TrustRoot::multi(&[new_vk.as_bytes(), old_vk.as_bytes()]).unwrap();
        let digest = [0xBBu8; 32];

        let new_sig = new_sk.sign(&digest);
        let old_sig = old_sk.sign(&digest);

        assert!(trust.verify(&digest, &new_sig.to_bytes()).is_ok());
        assert!(trust.verify(&digest, &old_sig.to_bytes()).is_ok());
    }

    #[test]
    fn rotation_window_rejects_signatures_from_third_party_key() {
        let (_, old_vk) = fixed_keypair(8);
        let (_, new_vk) = fixed_keypair(9);
        let (impostor_sk, _) = fixed_keypair(10);
        let trust = TrustRoot::multi(&[new_vk.as_bytes(), old_vk.as_bytes()]).unwrap();
        let digest = [0xCCu8; 32];
        let bad_sig = impostor_sk.sign(&digest);
        assert_eq!(
            trust.verify(&digest, &bad_sig.to_bytes()).unwrap_err(),
            SignatureError::Invalid
        );
    }

    #[test]
    fn empty_trust_root_rejects_everything() {
        let trust = TrustRoot::multi(&[]).unwrap();
        assert!(trust.is_empty());
        let (sk, _) = fixed_keypair(11);
        let sig = sk.sign(b"anything");
        assert_eq!(
            trust.verify(b"anything", &sig.to_bytes()).unwrap_err(),
            SignatureError::Invalid
        );
    }

    #[test]
    fn deterministic_seed_produces_consistent_pubkey() {
        // Sanity check: same seed → same verifying key. We rely on this for
        // test reproducibility.
        let (_, vk_a) = fixed_keypair(42);
        let (_, vk_b) = fixed_keypair(42);
        assert_eq!(vk_a.as_bytes(), vk_b.as_bytes());
    }
}
