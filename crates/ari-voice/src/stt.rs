//! Speech-to-text, moved from ari-android's `SpeechRecognizer.kt`. Two modes:
//!
//! - **Streaming** (Kroko, Nemotron: zipformer transducers): partials as the
//!   user speaks, partial-stability endpointing with a VAD veto, and a
//!   parallel stream as a second opinion. The English path.
//! - **Buffered** (Whisper Turbo, Parakeet Ultra, and the cloud): no partials,
//!   endpointing by loudness and the speech detector ([`BufferedEndpoint`]),
//!   one decode of the whole utterance at the end. The path for every language
//!   but English on Kroko.
//!
//! Each listen runs on a thread of its own, which owns its sherpa streams and
//! frees them itself. Its events go to the host through
//! [`VoiceHost::stt_event`](crate::pipeline::VoiceHost::stt_event).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::cloud_stt::{self, CloudSttFailure};
use crate::detector::SILERO_VAD_FILE;
use crate::endpoint::{
    BufferedEndpoint, ListenEnd, SPEECH_RMS_THRESHOLD, loudest_frame_rms, should_endpoint,
};
use crate::pipeline::{LogLevel, VoicePipeline};
use crate::sherpa::{OfflineModel, OfflineRecognizer, OnlineRecognizer, TransducerFiles, to_float};
use crate::speech_check::SpeechGate;
use crate::wake_phrase::{
    EarlyNameCheck, early_name_check, match_wake_phrase, streaming_partial, strip_wake_phrase, wake_verdict,
};

const SAMPLE_RATE: usize = 16_000;

/// Generous rewind: capture the full ring (~2 s) so device-power variations
/// in arm latency never clip the user's first word. The wake phrase that comes
/// along for the ride is stripped from the transcript, and endpoint detection
/// is gated on the stripped text, so the wake phrase never reaches the engine
/// and never causes a premature endpoint.
pub const DEFAULT_REWIND_SECONDS: f32 = 2.0;

/// 100 ms at 16 kHz: the chunk size sherpa was originally tuned against.
const BATCH_TARGET_SAMPLES: usize = 1_600;

/// 1 s at 16 kHz: the parallel stream's chunk. Bigger batches give the
/// decoder more context per acceptWaveform call.
const PARALLEL_BATCH_TARGET_SAMPLES: usize = 16_000;

/// How many main-stream batches to skip before feeding the parallel stream:
/// the pre-roll (silence and wake phrase), so the parallel decoder starts
/// clean on user speech only. 32000 pre-roll samples / 1600 = 20 batches.
const PREROLL_SKIP_BATCHES: usize = 20;

/// A model in a directory, as `SttModel` describes it.
#[derive(Debug, Clone)]
pub struct SttModelSpec {
    pub id: String,
    /// `zipformer`, `zipformer2` (streaming), `whisper` or `nemo_transducer` (buffered).
    pub model_type: String,
    pub dir: PathBuf,
    pub encoder_file: String,
    pub decoder_file: String,
    pub joiner_file: Option<String>,
    pub tokens_file: String,
}

/// How a buffered listen ended and how long after it started, for the wake log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListenEnded {
    pub end: ListenEnd,
    pub after_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SttError {
    /// No model is loaded and the cloud is not selected.
    NoModel,
    /// Something else holds the listen feed.
    Busy,
    /// The cloud request failed this way.
    Cloud(CloudSttFailure),
}

/// What a listen tells the host, in order: `Listening`, maybe `Transcribing`,
/// then one of the last three.
#[derive(Debug, Clone, PartialEq)]
pub enum SttEvent {
    /// `partial` is the transcript so far, wake phrase stripped; `name_matched`
    /// whether it holds the wake name. Only the streaming path has partials;
    /// the buffered paths send one empty one at the start.
    Listening { partial: String, name_matched: Option<bool> },
    /// Buffered paths: the user stopped, the decode is under way.
    Transcribing,
    /// `text` is the transcript with the wake phrase stripped, `parallel` the
    /// parallel stream's when it differs, `audio` the whole utterance, `raw`
    /// the transcript before stripping, `name_matched` whether `raw` held a
    /// real wake name (`None`: no verdict, taken as accept).
    Done {
        text: String,
        parallel: Option<String>,
        audio: Option<Vec<i16>>,
        raw: Option<String>,
        name_matched: Option<bool>,
        listen_end: Option<ListenEnded>,
    },
    /// Nothing was said: a quiet wake that heard no speech, or a cloud
    /// transcript that came back empty. `audio` is what was heard.
    NoSpeech { audio: Option<Vec<i16>>, listen_end: Option<ListenEnded> },
    Error(SttError),
}

/// A quiet wake's limits on a buffered listen ([`BufferedEndpoint`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ListenParams {
    pub rewind_seconds: f32,
    pub give_up_without_speech_ms: Option<i64>,
    pub limit_ms: Option<i64>,
}

impl Default for ListenParams {
    fn default() -> Self {
        Self { rewind_seconds: DEFAULT_REWIND_SECONDS, give_up_without_speech_ms: None, limit_ms: None }
    }
}

enum Loaded {
    Online(Arc<OnlineRecognizer>),
    /// Whisper's locale, which it bakes into its config; `None` for Parakeet,
    /// which finds the language itself.
    Offline(Arc<OfflineRecognizer>, Option<String>),
}

struct ListenHandle {
    id: u64,
    cancelled: Arc<AtomicBool>,
}

#[derive(Default)]
struct State {
    loaded: Option<Loaded>,
    model_id: Option<String>,
    cloud: bool,
    listen: Option<ListenHandle>,
    next_listen: u64,
}

/// The speech-to-text side of a [`VoicePipeline`].
#[derive(Default)]
pub struct Stt {
    state: Mutex<State>,
    /// Serialises loads, so a flurry of callers reacting to one setting change
    /// doesn't load the same ~1 GB model several times in parallel: each load
    /// instantiates a fresh ONNX runtime, and on Whisper Turbo that was enough
    /// memory pressure to get the wake word service killed.
    load_lock: Mutex<()>,
    /// The speech detector, built on first listen and reused: loading the
    /// model per utterance is far too expensive. A listen takes it and puts
    /// it back; a listen that finds it taken builds its own.
    gate: Mutex<Option<SpeechGate>>,
}

impl Stt {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn files_in(dir: &Path, spec: &SttModelSpec) -> Result<TransducerFiles, String> {
    let joiner = spec.joiner_file.as_ref().ok_or_else(|| format!("Transducer model {} has null joinerFile", spec.id))?;
    Ok(TransducerFiles {
        encoder: dir.join(&spec.encoder_file),
        decoder: dir.join(&spec.decoder_file),
        joiner: dir.join(joiner),
        tokens: dir.join(&spec.tokens_file),
    })
}

fn require_files(dir: &Path, files: &[&Path]) -> Result<(), String> {
    if files.iter().all(|f| f.is_file()) {
        Ok(())
    } else {
        Err(format!("Model files missing in {}", dir.display()))
    }
}

/// 16-bit mono 16 kHz PCM as a WAV file, for the cloud.
pub fn wav_bytes(pcm: &[i16]) -> Vec<u8> {
    let data = pcm.len() as u32 * 2;
    let mut out = Vec::with_capacity(44 + data as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE as u32).to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE as u32 * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

impl VoicePipeline {
    fn stt(&self) -> &Stt {
        &self.stt
    }

    fn info(&self, message: &str) {
        self.host().log(LogLevel::Info, message);
    }

    /// Route transcription to the cloud, or back to a local model. Releases
    /// any loaded recogniser on the way in: holding ~1 GB of ONNX for a path
    /// that will never call it is exactly the memory pressure that gets the
    /// wake word service killed.
    pub fn set_cloud_mode(&self, enabled: bool) {
        let mut state = self.stt().state();
        if state.cloud == enabled {
            return;
        }
        state.cloud = enabled;
        drop(state);
        if enabled {
            let _load = self.stt().load_lock.lock().unwrap_or_else(|e| e.into_inner());
            self.unload_stt();
            self.info("STT routed to cloud; local recogniser released");
        } else {
            self.info("STT routed back to on-device");
        }
    }

    pub fn cloud_mode(&self) -> bool {
        self.stt().state().cloud
    }

    /// A local model is loaded (the cloud needs none).
    pub fn stt_model_loaded(&self) -> bool {
        self.stt().state().loaded.is_some()
    }

    pub fn stt_model_id(&self) -> Option<String> {
        self.stt().state().model_id.clone()
    }

    /// The loaded recogniser is the streaming one and the cloud is not
    /// selected: partials are a usable liveness signal, and a retry decode
    /// means something.
    pub fn stt_streaming(&self) -> bool {
        let state = self.stt().state();
        !state.cloud && matches!(state.loaded, Some(Loaded::Online(_)))
    }

    fn loaded_with_matching_config(&self, spec: &SttModelSpec) -> bool {
        let state = self.stt().state();
        if state.model_id.as_deref() != Some(spec.id.as_str()) || state.loaded.is_none() {
            return false;
        }
        // Transducers have no per-locale config, so an id match is enough.
        // Whisper bakes the language into its config, so a locale change
        // needs a fresh load even when the model id matches.
        match &state.loaded {
            Some(Loaded::Offline(_, Some(locale))) => *locale == self.host().locale(),
            _ => true,
        }
    }

    /// Load `spec`, releasing any other model, and warm it up. Blocking: call
    /// it off the UI thread.
    pub fn load_stt(&self, spec: &SttModelSpec) -> Result<(), String> {
        // Fast path: no lock contention when the model is already loaded with
        // the right config.
        if self.loaded_with_matching_config(spec) {
            self.host().log(LogLevel::Debug, &format!("Model {} already loaded (fast path)", spec.id));
            return Ok(());
        }
        let _load = self.stt().load_lock.lock().unwrap_or_else(|e| e.into_inner());
        // Re-check inside the lock: another caller may have finished the load
        // meanwhile, and reloading would waste the ~3-10 s of warm-up just paid.
        if self.loaded_with_matching_config(spec) {
            self.host().log(LogLevel::Debug, &format!("Model {} already loaded (after lock)", spec.id));
            return Ok(());
        }
        self.unload_stt();
        let loaded = match spec.model_type.as_str() {
            "whisper" | "nemo_transducer" => self.load_offline(spec)?,
            "zipformer" | "zipformer2" => self.load_online(spec)?,
            other => return Err(format!("Unknown STT modelType: {other} (id={})", spec.id)),
        };
        let mut state = self.stt().state();
        state.loaded = Some(loaded);
        state.model_id = Some(spec.id.clone());
        Ok(())
    }

    fn load_online(&self, spec: &SttModelSpec) -> Result<Loaded, String> {
        let files = files_in(&spec.dir, spec)?;
        require_files(&spec.dir, &[&files.encoder, &files.decoder, &files.joiner, &files.tokens])?;
        self.info(&format!("Loading sherpa-onnx online model: {}", spec.id));
        let recognizer = OnlineRecognizer::new(&files, &spec.model_type)?;
        self.info(&format!("Online model loaded: {}", spec.id));
        // Warm up: the first decode on a fresh zipformer pays graph setup and
        // tensor arena allocation, easily 2-5 s on a phone CPU. Paid here, on
        // the loading thread, instead of on the first words after a wake,
        // which used to lose the first ~5 s of speech while the feed backed up.
        let started = std::time::Instant::now();
        let mut warmup = recognizer.create_stream()?;
        warmup.accept_waveform(&vec![0.0; SAMPLE_RATE / 5]); // 200 ms of silence
        warmup.decode_ready();
        drop(warmup);
        self.info(&format!("Online recogniser warmed in {}ms", started.elapsed().as_millis()));
        Ok(Loaded::Online(recognizer))
    }

    fn load_offline(&self, spec: &SttModelSpec) -> Result<Loaded, String> {
        let dir = &spec.dir;
        let encoder = dir.join(&spec.encoder_file);
        let decoder = dir.join(&spec.decoder_file);
        let tokens = dir.join(&spec.tokens_file);
        let joiner = spec.joiner_file.as_ref().map(|j| dir.join(j));
        let mut needed = vec![encoder.as_path(), decoder.as_path(), tokens.as_path()];
        if let Some(joiner) = &joiner {
            needed.push(joiner);
        }
        require_files(dir, &needed)?;
        let locale = self.host().locale();
        let whisper = spec.model_type == "whisper";
        let model = if whisper {
            OfflineModel::Whisper { encoder, decoder, tokens, language: locale.clone() }
        } else {
            OfflineModel::Transducer { files: files_in(dir, spec)?, model_type: spec.model_type.clone() }
        };
        self.info(&format!("Loading sherpa-onnx offline model: {} (lang={locale})", spec.id));
        let recognizer = OfflineRecognizer::new(&model)?;
        self.info(&format!("Offline model loaded: {}", spec.id));
        // Warm up: the first decode pays graph init and tensor arena
        // allocation, ~2-4 s on a phone CPU for Whisper. The result is
        // dropped; Whisper hallucinates on pure silence and nobody cares.
        let started = std::time::Instant::now();
        let _ = recognizer.transcribe(&vec![0.0; SAMPLE_RATE]); // 1 s of silence
        self.info(&format!("Offline recogniser warmed in {}ms", started.elapsed().as_millis()));
        Ok(Loaded::Offline(recognizer, whisper.then_some(locale)))
    }

    /// Release the model. A decode still running in a listen being torn down
    /// keeps its own reference, so the recogniser outlives it.
    pub fn unload_stt(&self) {
        self.stop_listen();
        let mut state = self.stt().state();
        state.loaded = None;
        state.model_id = None;
    }

    /// Start listening: the last `rewind_seconds` of the ring and what
    /// follows, through the cloud if selected, else the loaded model. Sends
    /// `Listening` at once, or an `Error`.
    pub fn start_listen(self: &Arc<Self>, params: ListenParams) {
        let (cloud, loaded) = {
            let state = self.stt().state();
            let loaded = match &state.loaded {
                Some(Loaded::Online(r)) => Some(Loaded::Online(Arc::clone(r))),
                Some(Loaded::Offline(r, l)) => Some(Loaded::Offline(Arc::clone(r), l.clone())),
                None => None,
            };
            (state.cloud, loaded)
        };
        // Cloud first: it needs no local model, so checking it last would let
        // a leftover recogniser from an earlier on-device session win after
        // the user switched to the cloud.
        let (label, kind) = match (cloud, loaded) {
            (true, _) => ("cloud", ListenKind::Cloud),
            (false, Some(Loaded::Online(r))) => ("online", ListenKind::Online(r)),
            (false, Some(Loaded::Offline(r, _))) => ("offline", ListenKind::Offline(r)),
            (false, None) => {
                self.host().log(LogLevel::Warn, "startListening called but no model loaded");
                self.host().stt_event(SttEvent::Error(SttError::NoModel));
                return;
            }
        };
        let Some(feed) = self.arm(params.rewind_seconds) else {
            self.host().log(LogLevel::Warn, "CaptureBus already armed — refusing to start listening");
            self.host().stt_event(SttEvent::Error(SttError::Busy));
            return;
        };
        self.host().stt_event(SttEvent::Listening { partial: String::new(), name_matched: Some(false) });
        self.info(&format!("STT ({label}) listening started (rewindSeconds={})", params.rewind_seconds));

        let cancelled = Arc::new(AtomicBool::new(false));
        let id = {
            let mut state = self.stt().state();
            state.next_listen += 1;
            let id = state.next_listen;
            if let Some(previous) = state.listen.replace(ListenHandle { id, cancelled: Arc::clone(&cancelled) }) {
                previous.cancelled.store(true, Ordering::Relaxed);
            }
            id
        };
        let pipeline = Arc::clone(self);
        let listen = Listen { pipeline, id, cancelled, feed, label, params };
        let spawned = std::thread::Builder::new().name("AriListen".into()).spawn(move || match kind {
            ListenKind::Online(recognizer) => listen.online(recognizer),
            ListenKind::Offline(recognizer) => listen.buffered(Transcriber::Offline(recognizer)),
            ListenKind::Cloud => listen.buffered(Transcriber::Cloud),
        });
        if spawned.is_err() {
            self.host().log(LogLevel::Warn, "could not start a listen thread");
            self.stop_listen_owned(Some(id));
        }
    }

    /// Stop whatever is listening and close its feed. Its sherpa streams are
    /// freed by its own thread on the way out, never here.
    pub fn stop_listen(&self) {
        self.stop_listen_owned(None);
    }

    /// A listen tearing down its own turn passes its id, and nothing happens
    /// if it is no longer the active listen: by the time a decode loop reaches
    /// its endpoint the user may have dismissed and re-armed, and an
    /// unconditional teardown would disarm the fresh one.
    fn stop_listen_owned(&self, owner: Option<u64>) {
        {
            let mut state = self.stt().state();
            if let Some(owner) = owner
                && state.listen.as_ref().is_none_or(|l| l.id != owner)
            {
                return;
            }
            if let Some(listen) = state.listen.take() {
                listen.cancelled.store(true, Ordering::Relaxed);
            }
        }
        self.disarm();
    }

    /// Decode `audio` again in a fresh streaming stream, all at once and
    /// finalised: the third-layer retry for a streaming transcript the engine
    /// did not understand. Slower, but the decoder sees the whole utterance
    /// before committing a token. `None` without a streaming model, or when
    /// nothing is left after the wake phrase. Blocking.
    pub fn transcribe_again(&self, audio: &[i16]) -> Option<String> {
        let recognizer = match &self.stt().state().loaded {
            Some(Loaded::Online(r)) => Arc::clone(r),
            _ => return None,
        };
        let mut stream = match recognizer.create_stream() {
            Ok(s) => s,
            Err(e) => {
                self.host().log(LogLevel::Warn, &format!("Offline retry failed: {e}"));
                return None;
            }
        };
        stream.accept_waveform(&to_float(audio));
        stream.input_finished();
        stream.decode_ready();
        let raw = stream.text().trim().to_string();
        let cleaned = strip_wake_phrase(&raw, &self.host().locale());
        self.info(&format!("Offline retry: raw='{raw}' cleaned='{cleaned}'"));
        (!cleaned.is_empty()).then_some(cleaned)
    }

    /// Whether the early name check can run: on cloud transcription only.
    pub fn can_check_name_early(&self) -> bool {
        self.cloud_mode()
    }

    /// Transcribe `pcm`, the 2 s ending at a quiet wake, on its own, and say
    /// whether it holds the wake name. The listen goes on meanwhile; this only
    /// lets the host show itself before the whole command is in. `None` where
    /// it cannot run (a self-hosted server: the check was only tested on
    /// OpenAI's). Blocking.
    pub fn check_name_early(&self, pcm: &[i16]) -> Option<EarlyNameCheck> {
        let locale = self.host().locale();
        let (service, key) = self.host().cloud_stt()?;
        match cloud_stt::transcribe_wake_clip(&service, key.as_deref(), &wav_bytes(pcm), &locale)? {
            Ok(transcript) => {
                let result = early_name_check(&transcript, &locale);
                self.info(&format!("Early name check: raw='{transcript}' -> {}", result.slug()));
                Some(result)
            }
            Err(e) => {
                self.info(&format!("Early name check: no transcript ({:?})", e.failure));
                Some(EarlyNameCheck::None)
            }
        }
    }

    fn obtain_gate(&self) -> Option<SpeechGate> {
        if let Some(gate) = self.stt().gate.lock().unwrap_or_else(|e| e.into_inner()).take() {
            return Some(gate);
        }
        // A gate we can't build is not fatal: the endpoint runs unvetoed and
        // the buffered path goes by loudness, as before the VAD was there.
        match self.host().load_model(SILERO_VAD_FILE).ok_or_else(|| format!("could not read {SILERO_VAD_FILE}")).and_then(|m| SpeechGate::new(&m)) {
            Ok(gate) => Some(gate),
            Err(e) => {
                self.host().log(
                    LogLevel::Warn,
                    &format!("VAD unavailable: listening by loudness alone, endpoint veto disabled: {e}"),
                );
                None
            }
        }
    }

    fn return_gate(&self, gate: Option<SpeechGate>) {
        if let Some(gate) = gate {
            self.stt().gate.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert(gate);
        }
    }
}

enum ListenKind {
    Online(Arc<OnlineRecognizer>),
    Offline(Arc<OfflineRecognizer>),
    Cloud,
}

enum Transcriber {
    Offline(Arc<OfflineRecognizer>),
    Cloud,
}

/// One listen, on its own thread.
struct Listen {
    pipeline: Arc<VoicePipeline>,
    id: u64,
    cancelled: Arc<AtomicBool>,
    feed: Receiver<Vec<i16>>,
    label: &'static str,
    params: ListenParams,
}

impl Listen {
    fn live(&self) -> bool {
        !self.cancelled.load(Ordering::Relaxed)
    }

    /// Hand `event` over unless this listen has been stopped meanwhile.
    fn emit(&self, event: SttEvent) {
        if self.live() {
            self.pipeline.host().stt_event(event);
        }
    }

    fn log(&self, level: LogLevel, message: &str) {
        self.pipeline.host().log(level, message);
    }

    fn finish(&self) {
        self.pipeline.stop_listen_owned(Some(self.id));
    }

    fn online(self, recognizer: Arc<OnlineRecognizer>) {
        // Both streams are created, used and freed on this thread alone and
        // never published anywhere. Two threads freeing one stream is a
        // double free: Bionic's allocator tolerates it, GrapheneOS's hardened
        // malloc aborts the process.
        let (mut current, mut parallel) = match (recognizer.create_stream(), recognizer.create_stream()) {
            (Ok(c), Ok(p)) => (c, p),
            _ => {
                self.log(LogLevel::Warn, "could not create the listen's streams");
                self.finish();
                return;
            }
        };
        self.log(LogLevel::Debug, "Streams created (main + parallel)");
        let mut gate = self.pipeline.obtain_gate();
        if let Some(g) = gate.as_mut() {
            g.begin_utterance();
        }
        let start = self.pipeline.now_ms();
        let mut first_chunk_logged = false;
        // sherpa's zipformer is tuned for ~100 ms chunks and the capture loop
        // writes 30 ms ones, so they are re-batched here. Without it sherpa
        // decodes three times per 100 ms and quality collapses ("tell me some
        // wisdom" became "Team").
        let mut batch: Vec<i16> = Vec::with_capacity(BATCH_TARGET_SAMPLES * 2);
        // The parallel stream: ~1 s batches of live audio only, no pre-roll.
        // The pre-roll feeds silence and the wake phrase into the main stream,
        // biasing its decoder ("Okay Ari,"). A different start state commits
        // different tokens: a meaningfully different second opinion.
        let mut parallel_batch: Vec<i16> = Vec::with_capacity(PARALLEL_BATCH_TARGET_SAMPLES * 2);
        let mut main_batches = 0;
        // Everything heard, for the retry decode.
        let mut audio: Vec<i16> = Vec::with_capacity(SAMPLE_RATE * 8);
        // The endpoint tracks when the cleaned partial last changed. Partial
        // stability is the least-bad signal there is, since sherpa's streaming
        // decoder lags the audio by an unpredictable amount on a phone; but
        // it is only ever a proxy for "the user stopped", so the speech
        // detector holds it back while it can still hear speech.
        let mut last_cleaned = String::new();
        let mut last_change_at = start;

        while self.live() {
            let Ok(incoming) = self.feed.recv() else { break };
            batch.extend_from_slice(&incoming);
            // Wait for a whole sherpa chunk. The pre-roll's slices are already
            // 1600 samples, so they flush at once.
            if batch.len() < BATCH_TARGET_SAMPLES {
                continue;
            }
            let merged = std::mem::take(&mut batch);
            if !first_chunk_logged {
                let after = self.pipeline.now_ms() - start;
                self.log(LogLevel::Info, &format!("First chunk decoded {after}ms after arm (size={})", merged.len()));
                first_chunk_logged = true;
            }
            audio.extend_from_slice(&merged);
            // One clock reading per batch, before the decode (which can burn
            // tens of ms): the VAD's speech times and the stability window
            // have to be measured on the same instant.
            let now = self.pipeline.now_ms();
            let floats = to_float(&merged);
            current.accept_waveform(&floats);
            if let Some(g) = gate.as_mut()
                && let Err(e) = g.feed(&floats, now)
            {
                self.log(LogLevel::Warn, &format!("VAD failed, endpoint veto disabled: {e}"));
                gate = None;
            }
            current.decode_ready();

            main_batches += 1;
            if main_batches > PREROLL_SKIP_BATCHES {
                parallel_batch.extend_from_slice(&merged);
                if parallel_batch.len() >= PARALLEL_BATCH_TARGET_SAMPLES {
                    parallel.accept_waveform(&to_float(&std::mem::take(&mut parallel_batch)));
                    parallel.decode_ready();
                }
            }

            let raw_partial = current.text().trim().to_string();
            let locale = self.pipeline.host().locale();
            let partial_match = match_wake_phrase(&raw_partial, &locale);
            let cleaned = partial_match.text.clone();
            self.log(
                LogLevel::Debug,
                &format!("decode: fed={} raw='{raw_partial}' cleaned='{cleaned}'", merged.len()),
            );
            if let Some(p) = streaming_partial(&partial_match, &locale) {
                self.emit(SttEvent::Listening { partial: p.text, name_matched: p.name_matched });
            }
            if cleaned != last_cleaned {
                last_cleaned = cleaned.clone();
                last_change_at = now;
            }
            // Zero stability while the partial is empty, so it can't satisfy
            // the stability arm; the check still runs, because the 30 s cap is
            // the only way out when noise keeps the VAD hot and the decoder
            // never settles.
            let stable_for = if cleaned.is_empty() { 0 } else { now - last_change_at };
            let since_speech = gate.as_ref().map_or(i64::MAX, |g| g.ms_since_last_speech(now));
            if !should_endpoint(stable_for, since_speech, now - start) {
                continue;
            }
            let since = if since_speech == i64::MAX { "never".to_string() } else { format!("{since_speech}ms") };
            self.log(
                LogLevel::Info,
                &format!(
                    "Custom endpoint: stable={stable_for}ms sinceSpeech={since} listening={}ms cleaned='{cleaned}'",
                    now - start
                ),
            );
            // Finalise the main stream before reading it. sherpa holds tokens
            // back pending right context, and the endpoint fires exactly when
            // the partial has gone quiet, which is when those tokens are still
            // owed. Reading without input_finished() discards them: "turn on
            // the kitchen table light" arrived as "ton". Greedy transducer
            // decoding cannot retract a committed token, so this only appends.
            current.input_finished();
            current.decode_ready();
            let flushed = current.text().trim().to_string();
            let (final_raw, final_match) = if flushed.is_empty() {
                (raw_partial, partial_match)
            } else {
                let m = match_wake_phrase(&flushed, &locale);
                (flushed, m)
            };
            if final_match.text != cleaned {
                self.log(LogLevel::Info, &format!("Flush recovered: '{cleaned}' -> '{}'", final_match.text));
            }
            // The rest of the audio into the parallel stream, then finalise it
            // for the not-understood retry.
            if !parallel_batch.is_empty() {
                parallel.accept_waveform(&to_float(&std::mem::take(&mut parallel_batch)));
            }
            parallel.input_finished();
            parallel.decode_ready();
            let parallel_raw = parallel.text().trim().to_string();
            let parallel_cleaned = strip_wake_phrase(&parallel_raw, &locale);
            self.log(
                LogLevel::Info,
                &format!("Parallel stream final: raw='{parallel_raw}' cleaned='{parallel_cleaned}'"),
            );
            let parallel_text =
                (!parallel_cleaned.is_empty() && parallel_cleaned != final_match.text).then_some(parallel_cleaned);
            self.emit(SttEvent::Done {
                name_matched: wake_verdict(&final_match, &locale),
                text: final_match.text,
                parallel: parallel_text,
                audio: Some(audio),
                raw: Some(final_raw),
                listen_end: None,
            });
            break;
        }
        drop(current);
        drop(parallel);
        self.pipeline.return_gate(gate);
        self.finish();
    }

    /// Buffer until the user stops talking, then transcribe the whole
    /// utterance. Shared by Whisper, Parakeet and the cloud, because only
    /// where the decode happens differs: duplicating the capture, the
    /// endpointing and the cap per backend is how the rules drift apart.
    fn buffered(self, transcriber: Transcriber) {
        let mut audio: Vec<i16> = Vec::with_capacity(SAMPLE_RATE * 8);
        let mut gate = self.pipeline.obtain_gate();
        if let Some(g) = gate.as_mut() {
            g.begin_utterance();
        }
        let started_at = self.pipeline.now_ms();
        let mut endpoint =
            BufferedEndpoint::new(started_at, self.params.give_up_without_speech_ms, self.params.limit_ms);
        let mut ended = None;
        while self.live() {
            let Ok(incoming) = self.feed.recv() else { break };
            if incoming.is_empty() {
                continue;
            }
            audio.extend_from_slice(&incoming);
            let now = self.pipeline.now_ms();
            let loud = loudest_frame_rms(&incoming) >= SPEECH_RMS_THRESHOLD;
            let heard = match gate.as_mut().map(|g| g.heard_speech(&incoming)) {
                Some(Ok(heard)) => heard,
                Some(Err(e)) => {
                    // Carry on by loudness alone, as before the detector was here.
                    self.log(LogLevel::Warn, &format!("{}: speech detector failed, listening by loudness alone: {e}", self.label));
                    gate = None;
                    false
                }
                None => false,
            };
            if let Some(end) = endpoint.next(now, loud, heard, audio.len()) {
                let after_ms = now - started_at;
                self.log(
                    LogLevel::Info,
                    &format!("{} listen ended: {} after {after_ms}ms ({} samples)", self.label, end.slug(), audio.len()),
                );
                ended = Some(ListenEnded { end, after_ms });
                break;
            }
        }
        self.pipeline.return_gate(gate);
        if let Some(listen_end) = ended {
            if listen_end.end == ListenEnd::GiveUp {
                self.emit(SttEvent::NoSpeech { audio: Some(audio), listen_end: Some(listen_end) });
            } else {
                match transcriber {
                    Transcriber::Offline(recognizer) => self.decode_offline(&recognizer, audio, listen_end),
                    Transcriber::Cloud => self.transcribe_cloud(audio, listen_end),
                }
            }
        }
        self.finish();
    }

    fn decode_offline(&self, recognizer: &Arc<OfflineRecognizer>, audio: Vec<i16>, listen_end: ListenEnded) {
        // Posted before the decode, so the host can show it is thinking
        // rather than still listening for the whole decode window.
        self.emit(SttEvent::Transcribing);
        let transcript = match recognizer.transcribe(&to_float(&audio)) {
            Ok(text) => text.trim().to_string(),
            Err(e) => {
                self.log(LogLevel::Warn, &format!("Offline decode failed: {e}"));
                String::new()
            }
        };
        let locale = self.pipeline.host().locale();
        let m = match_wake_phrase(&transcript, &locale);
        self.log(LogLevel::Info, &format!("Offline transcript: raw='{transcript}' cleaned='{}'", m.text));
        // No parallel stream to disagree with; the audio still goes along, so
        // the host can keep it for the debug captures.
        self.emit(SttEvent::Done {
            name_matched: wake_verdict(&m, &locale),
            text: m.text,
            parallel: None,
            audio: Some(audio),
            raw: Some(transcript),
            listen_end: Some(listen_end),
        });
    }

    fn transcribe_cloud(&self, audio: Vec<i16>, listen_end: ListenEnded) {
        self.emit(SttEvent::Transcribing);
        let locale = self.pipeline.host().locale();
        let Some((service, key)) = self.pipeline.host().cloud_stt() else {
            self.emit(SttEvent::Error(SttError::Cloud(CloudSttFailure::NotConfigured)));
            return;
        };
        match cloud_stt::transcribe(&service, key.as_deref(), &wav_bytes(&audio), &locale) {
            Ok(transcript) => {
                let m = match_wake_phrase(&transcript, &locale);
                self.log(LogLevel::Info, &format!("Cloud transcript: raw='{transcript}' cleaned='{}'", m.text));
                // No second decoder and no local model to re-run: a cloud retry
                // would be the same request billed twice.
                self.emit(SttEvent::Done {
                    name_matched: wake_verdict(&m, &locale),
                    text: m.text,
                    parallel: None,
                    audio: Some(audio),
                    raw: Some(transcript),
                    listen_end: Some(listen_end),
                });
            }
            Err(e) => {
                self.log(LogLevel::Warn, &format!("Cloud STT failed ({:?}): {}", e.failure, e.message));
                // Hearing nothing is not a failed request, and on a wake turn it
                // is the commonest false wake there is: the host needs the audio
                // to keep it and the distinction to count it.
                if e.failure == CloudSttFailure::Empty {
                    self.emit(SttEvent::NoSpeech { audio: Some(audio), listen_end: Some(listen_end) });
                } else {
                    self.emit(SttEvent::Error(SttError::Cloud(e.failure)));
                }
            }
        }
    }
}
