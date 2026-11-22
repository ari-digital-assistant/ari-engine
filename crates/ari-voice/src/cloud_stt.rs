//! Speech-to-text against any OpenAI-compatible `/audio/transcriptions`
//! endpoint.
//!
//! Deliberately not "the OpenAI transcriber". The same request shape is served
//! by OpenAI, by Groq, and by self-hosted `whisper.cpp` / `faster-whisper`,
//! including the Whisper add-on that ships with Home Assistant's Assist stack.
//! So the endpoint is a user-supplied URL and the API key is optional: point it
//! at your own box and no key is needed, no audio leaves your network, and
//! there is no per-minute bill. That is the difference between offering a cloud
//! option and taking a dependency on one vendor.
//!
//! Blocking: callers run it off the main thread.

use std::sync::Arc;
use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::CONTENT_TYPE;

/// The endpoint for [`CloudSttService::OpenAi`]. Not user-editable: that is the
/// preset.
pub const OPENAI_ENDPOINT: &str = "https://api.openai.com/v1";

/// OpenAI's recommended transcription model (verified against their API docs,
/// 2026-08). NOT `whisper-1`: that is now labelled legacy and retained mainly
/// for word/segment timestamps and English translation, and it has a worse word
/// error rate. Shipping the legacy model as the "more accurate than on-device"
/// option would defeat the purpose.
pub const OPENAI_MODEL: &str = "gpt-transcribe";

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Generous: a self-hosted large-v3 on modest hardware is slower than a hosted
/// API, and a timeout here costs the user their whole utterance. The Android
/// client allowed 60 s per read after a 10 s connect; reqwest's blocking client
/// has no per-read timeout, so the whole request gets both.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(70);

/// Where a cloud transcription attempt went wrong, so the host can say why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudSttFailure {
    /// No endpoint, or the OpenAI preset with no key: a Settings trip.
    NotConfigured,
    Network,
    /// The key was rejected: a new key, not a Settings trip.
    Auth,
    Server,
    /// The server answered with no words. "Ari heard nothing" and "the request
    /// failed" want different words in front of the user.
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudSttError {
    pub failure: CloudSttFailure,
    pub message: String,
}

impl CloudSttError {
    fn new(failure: CloudSttFailure, message: impl Into<String>) -> Self {
        Self {
            failure,
            message: message.into(),
        }
    }
}

/// Which service to send audio to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloudSttService {
    /// We own the endpoint and the model, so the user only has to supply a key.
    OpenAi,
    /// The opposite: we know neither.
    SelfHosted { endpoint: String, model: String },
}

/// Transcribe `wav` (16-bit mono 16 kHz) and return the text.
///
/// Never returns blank: an empty transcript is [`CloudSttFailure::Empty`].
pub fn transcribe(
    service: &CloudSttService,
    api_key: Option<&str>,
    wav: &[u8],
    locale: &str,
) -> Result<String, CloudSttError> {
    let (base, model) = match service {
        CloudSttService::OpenAi => (OPENAI_ENDPOINT, OPENAI_MODEL),
        CloudSttService::SelfHosted { endpoint, model } => (endpoint.as_str(), model.as_str()),
    };
    if base.trim().is_empty() {
        return Err(CloudSttError::new(
            CloudSttFailure::NotConfigured,
            "no cloud STT endpoint configured",
        ));
    }
    let api_key = api_key.filter(|k| !k.trim().is_empty());
    if *service == CloudSttService::OpenAi && api_key.is_none() {
        // Distinguish "you haven't set this up" from "your key was rejected":
        // the first is a Settings trip, the second is a new key.
        return Err(CloudSttError::new(
            CloudSttFailure::NotConfigured,
            "no OpenAI API key set",
        ));
    }

    let boundary = format!("----AriBoundary{}", wav.len());
    let body = multipart_body(wav, model, locale, &boundary);
    let mut request = http_client()?
        .post(transcription_url(base))
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body);
    // Only when the user gave us one: a self-hosted endpoint has no key, and
    // sending an empty Authorization header makes some servers reject the
    // request outright.
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    let response = request.send().map_err(network)?;
    let status = response.status();
    if !status.is_success() {
        let detail: String = response
            .text()
            .unwrap_or_default()
            .chars()
            .take(300)
            .collect();
        return Err(CloudSttError::new(
            failure_for(status.as_u16()),
            format!("HTTP {}: {detail}", status.as_u16()),
        ));
    }
    let text = parse_transcript(&response.text().map_err(network)?);
    if text.is_empty() {
        return Err(CloudSttError::new(
            CloudSttFailure::Empty,
            "empty transcript",
        ));
    }
    Ok(text)
}

/// [`transcribe`], for the short clip a quiet wake checks for its name. OpenAI
/// only, the one service the early check was tested on: `None` on a
/// self-hosted server.
pub fn transcribe_wake_clip(
    service: &CloudSttService,
    api_key: Option<&str>,
    wav: &[u8],
    locale: &str,
) -> Option<Result<String, CloudSttError>> {
    match service {
        CloudSttService::OpenAi => Some(transcribe(service, api_key, wav, locale)),
        CloudSttService::SelfHosted { .. } => None,
    }
}

fn network(e: reqwest::Error) -> CloudSttError {
    CloudSttError::new(CloudSttFailure::Network, format!("request failed: {e}"))
}

/// A client trusting the bundled webpki-roots CA set, as every other reqwest
/// client in the engine does. reqwest's own `rustls` setup goes through
/// `rustls-platform-verifier`, which needs JNI initialisation on Android before
/// its first handshake; `ari-skill-loader`'s `tls.rs` has the full story.
///
/// The `ring` provider is pinned because two providers are linked into the
/// binary and rustls refuses to pick between them.
fn http_client() -> Result<Client, CloudSttError> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let tls = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .expect("rustls ring provider supports TLS 1.2 + 1.3")
    .with_root_certificates(roots)
    .with_no_client_auth();
    Client::builder()
        .use_preconfigured_tls(tls)
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(network)
}

/// Resolve the transcription URL from whatever the user typed. Accepts a bare
/// host, an OpenAI-style `/v1` base, or the full path: people paste all three,
/// and a 404 with no explanation is a miserable way to find out which one was
/// wanted.
pub fn transcription_url(base: &str) -> String {
    let trimmed = base.trim().trim_end_matches('/');
    if trimmed.ends_with("/audio/transcriptions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/audio/transcriptions")
    }
}

/// Map an HTTP status onto something we can say out loud.
pub fn failure_for(status: u16) -> CloudSttFailure {
    match status {
        401 | 403 => CloudSttFailure::Auth,
        _ => CloudSttFailure::Server,
    }
}

/// Pull the transcript out of a response. OpenAI-compatible servers answer
/// `{"text": "..."}`; a plain-text body is tolerated because some self-hosted
/// builds return exactly that.
pub fn parse_transcript(body: &str) -> String {
    let trimmed = body.trim();
    if !trimmed.starts_with('{') {
        return trimmed.to_string();
    }
    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(json) => json
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .trim()
            .to_string(),
        Err(_) => String::new(),
    }
}

/// Build the multipart body. `locale` is sent as `language` so the server does
/// not have to guess: Whisper's auto-detection is good but it does misfire on
/// short commands, which is all Ari ever sends.
pub fn multipart_body(wav: &[u8], model: &str, locale: &str, boundary: &str) -> Vec<u8> {
    // Two-letter code only: the API wants ISO-639-1, so "en-GB" is rejected
    // where "en" is fine.
    let language = locale.chars().take(2).collect::<String>().to_lowercase();
    let mut out = Vec::with_capacity(wav.len() + 512);
    for (name, value) in [("model", model), ("language", language.as_str())] {
        out.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    out.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n\
             Content-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    out.extend_from_slice(wav);
    out.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::thread;

    // URL resolution: people paste all three of these.

    #[test]
    fn openai_style_base_gets_the_path_appended() {
        assert_eq!(
            transcription_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/audio/transcriptions"
        );
    }

    #[test]
    fn full_path_is_left_alone() {
        assert_eq!(
            transcription_url("http://homeassistant.local:10300/v1/audio/transcriptions"),
            "http://homeassistant.local:10300/v1/audio/transcriptions"
        );
    }

    #[test]
    fn trailing_slash_and_whitespace_are_tolerated() {
        assert_eq!(
            transcription_url("  https://api.openai.com/v1/  "),
            "https://api.openai.com/v1/audio/transcriptions"
        );
    }

    // Response parsing.

    #[test]
    fn json_body_yields_text_field() {
        assert_eq!(
            parse_transcript(r#"{"text":"turn on the kitchen light"}"#),
            "turn on the kitchen light"
        );
    }

    #[test]
    fn json_text_is_trimmed() {
        assert_eq!(parse_transcript(r#"{"text":"  hello  "}"#), "hello");
    }

    #[test]
    fn plain_text_body_is_accepted() {
        // Some self-hosted whisper builds answer with the bare transcript.
        assert_eq!(parse_transcript("how's the weather\n"), "how's the weather");
    }

    #[test]
    fn json_without_text_field_is_blank() {
        assert_eq!(parse_transcript(r#"{"error":"nope"}"#), "");
    }

    #[test]
    fn malformed_json_is_blank_not_a_crash() {
        assert_eq!(parse_transcript(r#"{"text": "#), "");
    }

    // Status mapping.

    #[test]
    fn unauthorized_and_forbidden_are_auth_failures() {
        assert_eq!(failure_for(401), CloudSttFailure::Auth);
        assert_eq!(failure_for(403), CloudSttFailure::Auth);
    }

    #[test]
    fn other_errors_are_server_failures() {
        assert_eq!(failure_for(500), CloudSttFailure::Server);
        assert_eq!(failure_for(429), CloudSttFailure::Server);
        assert_eq!(failure_for(404), CloudSttFailure::Server);
    }

    // Multipart body.

    fn body_text(body: &[u8]) -> String {
        body.iter().map(|b| *b as char).collect()
    }

    #[test]
    fn multipart_carries_model_language_and_wav() {
        let body = body_text(&multipart_body(b"RIFF", "whisper-1", "en", "BOUND"));
        assert_eq!(
            body,
            "--BOUND\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n\
             --BOUND\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nen\r\n\
             --BOUND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n\
             Content-Type: audio/wav\r\n\r\nRIFF\r\n--BOUND--\r\n"
        );
    }

    #[test]
    fn regional_locale_is_narrowed_to_iso_639_1() {
        // The API rejects "en-GB"; it wants "en".
        let body = body_text(&multipart_body(&[], "whisper-1", "en-GB", "B"));
        assert!(body.contains("name=\"language\"\r\n\r\nen\r\n"), "{body}");
    }

    #[test]
    fn uppercase_locale_is_lowercased() {
        let body = body_text(&multipart_body(&[], "whisper-1", "IT", "B"));
        assert!(body.contains("name=\"language\"\r\n\r\nit\r\n"), "{body}");
    }

    #[test]
    fn the_openai_preset_is_the_current_model_not_the_legacy_one() {
        assert_eq!(OPENAI_MODEL, "gpt-transcribe");
        assert_eq!(OPENAI_ENDPOINT, "https://api.openai.com/v1");
    }

    // Requests, against a one-shot server on localhost.

    fn self_hosted(endpoint: &str) -> CloudSttService {
        CloudSttService::SelfHosted {
            endpoint: endpoint.into(),
            model: "whisper-1".into(),
        }
    }

    /// Serves one request with `status` and `body`, and returns the request's
    /// head and body as it arrived.
    fn serve_once(
        status: &'static str,
        body: &'static str,
    ) -> (String, thread::JoinHandle<(String, Vec<u8>)>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut head = String::new();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                head.push_str(&line);
            }
            let length: usize = head
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length: ")
                        .map(|v| v.trim().parse().unwrap())
                })
                .unwrap_or(0);
            let mut request_body = vec![0; length];
            reader.read_exact(&mut request_body).unwrap();
            let mut stream = stream;
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            (head, request_body)
        });
        (endpoint, handle)
    }

    #[test]
    fn a_self_hosted_server_over_plain_http_is_sent_the_audio_without_a_key() {
        let (endpoint, server) = serve_once("200 OK", r#"{"text":" what time is it "}"#);

        let text = transcribe(&self_hosted(&endpoint), Some("  "), b"RIFFdata", "en-GB").unwrap();

        assert_eq!(text, "what time is it");
        let (head, body) = server.join().unwrap();
        assert!(
            head.starts_with("POST /v1/audio/transcriptions HTTP/1.1\r\n"),
            "{head}"
        );
        assert!(
            !head.to_ascii_lowercase().contains("authorization"),
            "{head}"
        );
        assert!(
            head.contains("multipart/form-data; boundary=----AriBoundary8"),
            "{head}"
        );
        assert_eq!(
            body,
            multipart_body(b"RIFFdata", "whisper-1", "en-GB", "----AriBoundary8")
        );
    }

    #[test]
    fn a_key_is_sent_as_a_bearer_token() {
        let (endpoint, server) = serve_once("200 OK", "hello");

        assert_eq!(
            transcribe(&self_hosted(&endpoint), Some("sk-test"), b"", "en").unwrap(),
            "hello"
        );
        let (head, _) = server.join().unwrap();
        assert!(
            head.to_ascii_lowercase()
                .contains("authorization: bearer sk-test\r\n"),
            "{head}"
        );
    }

    #[test]
    fn a_rejected_key_is_an_auth_failure_with_the_servers_reason() {
        let (endpoint, server) = serve_once("401 Unauthorized", r#"{"error":"bad key"}"#);

        let err = transcribe(&self_hosted(&endpoint), Some("sk-test"), b"", "en").unwrap_err();

        assert_eq!(
            err,
            CloudSttError::new(CloudSttFailure::Auth, r#"HTTP 401: {"error":"bad key"}"#)
        );
        server.join().unwrap();
    }

    #[test]
    fn a_server_error_is_a_server_failure() {
        let (endpoint, server) = serve_once("500 Internal Server Error", "");

        let err = transcribe(&self_hosted(&endpoint), None, b"", "en").unwrap_err();

        assert_eq!(
            err,
            CloudSttError::new(CloudSttFailure::Server, "HTTP 500: ")
        );
        server.join().unwrap();
    }

    #[test]
    fn a_blank_transcript_is_empty_not_success() {
        let (endpoint, server) = serve_once("200 OK", r#"{"text":"   "}"#);

        let err = transcribe(&self_hosted(&endpoint), None, b"", "en").unwrap_err();

        assert_eq!(err.failure, CloudSttFailure::Empty);
        server.join().unwrap();
    }

    #[test]
    fn nobody_listening_is_a_network_failure() {
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();

        let err = transcribe(
            &self_hosted(&format!("http://127.0.0.1:{port}")),
            None,
            b"",
            "en",
        )
        .unwrap_err();

        assert_eq!(err.failure, CloudSttFailure::Network);
    }

    #[test]
    fn no_endpoint_is_not_configured() {
        let err = transcribe(&self_hosted("   "), None, b"", "en").unwrap_err();
        assert_eq!(
            err,
            CloudSttError::new(
                CloudSttFailure::NotConfigured,
                "no cloud STT endpoint configured"
            )
        );
    }

    #[test]
    fn the_openai_preset_without_a_key_is_not_configured_and_sends_nothing() {
        for key in [None, Some(""), Some("  ")] {
            let err = transcribe(&CloudSttService::OpenAi, key, b"", "en").unwrap_err();
            assert_eq!(
                err,
                CloudSttError::new(CloudSttFailure::NotConfigured, "no OpenAI API key set")
            );
        }
    }

    #[test]
    fn the_wake_clip_check_runs_on_openai_only() {
        assert_eq!(
            transcribe_wake_clip(&self_hosted("http://127.0.0.1:9"), None, b"", "en"),
            None
        );
        assert_eq!(
            transcribe_wake_clip(&CloudSttService::OpenAi, None, b"", "en"),
            Some(Err(CloudSttError::new(
                CloudSttFailure::NotConfigured,
                "no OpenAI API key set"
            ))),
        );
    }
}
