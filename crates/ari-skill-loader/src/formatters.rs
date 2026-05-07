//! Locale-aware date / number / currency formatters used by the
//! `ari::format_date`, `ari::format_number`, and `ari::format_currency`
//! WASM host imports.
//!
//! Hand-rolled lookup tables for the 5 supported locales (`en`, `it`,
//! `es`, `fr`, `de`) rather than pulling in the full `icu` / `icu4x`
//! crate stack. Trade-off:
//!
//! - **Hand-rolled (this module):** ~3 KB of binary impact, covers the
//!   languages we plan to support in the Phase-3 horizon, easy to add a
//!   new locale (one entry in [`LOCALES`]). Doesn't handle plural
//!   forms, relative dates ("3 days ago"), CJK, or the long tail of
//!   Unicode tailorings.
//! - **`icu4x`:** ~10–15 MB binary impact (CLDR data), full Unicode
//!   correctness, covers ~700 locales. Worth swapping in when we
//!   support more than ~10 languages or when we need plurals / relative
//!   dates / etc.
//!
//! The host-import surface (`format_date(ts_ms, locale, style)` etc.)
//! is identical regardless of which backend is in play, so the swap is
//! a self-contained internal change when it happens.
//!
//! ## Locale fallback
//!
//! [`format_for`] resolves an unknown locale to the canonical English
//! formatter — same fallback rule as the rest of the multi-language
//! plumbing. A skill passing `"jp"` for a Japanese-locale request gets
//! English output rather than an error; the host's locale provider
//! should never produce `"jp"` for a real user, but the safety net is
//! cheap and keeps the host-import contract total.

use crate::platform_capabilities::days_to_ymd;

/// Style hint accepted by the host-import functions. Hand-rolled
/// implementation only differentiates `Long` from everything else for
/// dates (full month name vs short); kept on the API for forward
/// compatibility with an eventual `icu4x` backend that respects all
/// four levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatStyle {
    Short,
    Medium,
    Long,
    Full,
}

impl FormatStyle {
    pub fn parse(s: &str) -> Self {
        match s {
            "short" => FormatStyle::Short,
            "long" => FormatStyle::Long,
            "full" => FormatStyle::Full,
            // Anything else (including `""` from a default-args call
            // and unknown values) defaults to medium — the most
            // commonly-useful choice for a skill that doesn't care.
            _ => FormatStyle::Medium,
        }
    }
}

/// Per-locale formatting data. One entry per supported language code.
struct LocaleFormatter {
    /// ISO 639-1 lowercase code this entry serves.
    code: &'static str,
    /// Full month names, January..December (index 0..=11).
    month_full: [&'static str; 12],
    /// Short month names, Jan..Dec.
    month_short: [&'static str; 12],
    /// Decimal point in numbers (`'.'` for en, `','` for it/de/es/fr).
    decimal: char,
    /// Group separator for thousands. `' '` for fr (a U+00A0 non-
    /// breaking space in real CLDR data; ASCII space is good enough
    /// for our purposes).
    group: char,
    /// Where the currency symbol goes relative to the amount.
    currency_position: CurrencyPosition,
    /// Whether the currency symbol is separated from the amount by a
    /// space.
    currency_space: bool,
}

#[derive(Debug, Clone, Copy)]
enum CurrencyPosition {
    Prefix,
    Suffix,
}

const LOCALES: &[LocaleFormatter] = &[
    LocaleFormatter {
        code: "en",
        month_full: [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        month_short: [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ],
        decimal: '.',
        group: ',',
        currency_position: CurrencyPosition::Prefix,
        currency_space: false,
    },
    LocaleFormatter {
        code: "it",
        month_full: [
            "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
            "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
        ],
        month_short: [
            "gen", "feb", "mar", "apr", "mag", "giu",
            "lug", "ago", "set", "ott", "nov", "dic",
        ],
        decimal: ',',
        group: '.',
        currency_position: CurrencyPosition::Suffix,
        currency_space: true,
    },
    LocaleFormatter {
        code: "es",
        month_full: [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        ],
        month_short: [
            "ene", "feb", "mar", "abr", "may", "jun",
            "jul", "ago", "sep", "oct", "nov", "dic",
        ],
        decimal: ',',
        group: '.',
        currency_position: CurrencyPosition::Suffix,
        currency_space: true,
    },
    LocaleFormatter {
        code: "fr",
        month_full: [
            "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre",
        ],
        month_short: [
            "janv.", "févr.", "mars", "avr.", "mai", "juin",
            "juil.", "août", "sept.", "oct.", "nov.", "déc.",
        ],
        decimal: ',',
        group: ' ',
        currency_position: CurrencyPosition::Suffix,
        currency_space: true,
    },
    LocaleFormatter {
        code: "de",
        month_full: [
            "Januar", "Februar", "März", "April", "Mai", "Juni",
            "Juli", "August", "September", "Oktober", "November", "Dezember",
        ],
        month_short: [
            "Jan.", "Feb.", "März", "Apr.", "Mai", "Juni",
            "Juli", "Aug.", "Sept.", "Okt.", "Nov.", "Dez.",
        ],
        decimal: ',',
        group: '.',
        currency_position: CurrencyPosition::Suffix,
        currency_space: true,
    },
];

/// Resolve a locale code to its formatter, falling back to canonical
/// English when the requested locale isn't in the supported set.
fn format_for(locale: &str) -> &'static LocaleFormatter {
    LOCALES
        .iter()
        .find(|f| f.code == locale)
        .unwrap_or(&LOCALES[0])
}

/// Format a Unix epoch millisecond timestamp as a date in the given
/// locale. Style controls month-name length:
///
/// - `Short` / `Medium` → "30 Apr 2026" (en) / "30 apr 2026" (it)
/// - `Long` / `Full` → "30 April 2026" (en) / "30 aprile 2026" (it)
///
/// For all locales the day-month-year order is the same — no locale
/// in our supported set inverts to month-day-year (US-style "April 30,
/// 2026" is intentionally NOT special-cased; we ship the locale-
/// neutral DMY form because Ari users in the US still parse "30
/// April" without confusion, and the alternative requires a separate
/// `en-US` formatter).
pub fn format_date(ts_ms: i64, locale: &str, style: FormatStyle) -> String {
    let fmt = format_for(locale);
    let day_secs = ts_ms.div_euclid(86_400_000);
    let (year, month, day) = days_to_ymd(day_secs);
    let month_idx = (month as usize).saturating_sub(1).min(11);
    let month_str = match style {
        FormatStyle::Long | FormatStyle::Full => fmt.month_full[month_idx],
        _ => fmt.month_short[month_idx],
    };
    format!("{} {} {}", day, month_str, year)
}

/// Format a floating-point number using the locale's decimal and
/// group separators. Two decimal places by default — calling skills
/// that need different precision should round before passing in. We
/// don't expose precision on the host import yet (style argument is
/// reserved for it).
///
/// Negative numbers get a leading minus sign with no spacing
/// (`"-1.234,56"`); we don't support locale-specific negative
/// patterns (parentheses, brackets) — adds complexity for marginal
/// gain in our supported set.
pub fn format_number(value: f64, locale: &str, _style: FormatStyle) -> String {
    let fmt = format_for(locale);
    format_decimal(value, fmt.decimal, fmt.group, 2)
}

/// Format a monetary amount with a currency symbol or code, in the
/// locale's customary position. Currency-code → symbol resolution is
/// a small lookup table covering the major currencies — unknown
/// codes pass through verbatim ("XAU 1.234,56" rather than failing).
pub fn format_currency(amount: f64, currency_code: &str, locale: &str) -> String {
    let fmt = format_for(locale);
    let symbol = currency_symbol(currency_code);
    let amount_str = format_decimal(amount, fmt.decimal, fmt.group, 2);
    let sep = if fmt.currency_space { " " } else { "" };
    match fmt.currency_position {
        CurrencyPosition::Prefix => format!("{symbol}{sep}{amount_str}"),
        CurrencyPosition::Suffix => format!("{amount_str}{sep}{symbol}"),
    }
}

fn currency_symbol(code: &str) -> &str {
    match code {
        "USD" => "$",
        "EUR" => "€",
        "GBP" => "£",
        "JPY" | "CNY" => "¥",
        "CHF" => "Fr.",
        "AUD" => "A$",
        "CAD" => "C$",
        "INR" => "₹",
        "KRW" => "₩",
        "RUB" => "₽",
        // Unknown currency: pass the code through verbatim. Better
        // than swallowing it and producing "1.234,56" with no
        // currency context.
        _ => code,
    }
}

fn format_decimal(value: f64, decimal: char, group: char, precision: usize) -> String {
    let negative = value.is_sign_negative();
    let abs = value.abs();
    // Round to `precision` decimals before splitting so we don't
    // accumulate float error in the integer part.
    let rounded = (abs * 10f64.powi(precision as i32)).round() / 10f64.powi(precision as i32);
    let raw = format!("{rounded:.precision$}");
    let (int_part, frac_part) = match raw.find('.') {
        Some(i) => (&raw[..i], &raw[i + 1..]),
        None => (raw.as_str(), ""),
    };
    // Group the integer part right-to-left.
    let mut grouped = String::with_capacity(int_part.len() + int_part.len() / 3);
    let chars: Vec<char> = int_part.chars().rev().collect();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && i % 3 == 0 {
            grouped.push(group);
        }
        grouped.push(*c);
    }
    let int_grouped: String = grouped.chars().rev().collect();
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    out.push_str(&int_grouped);
    if precision > 0 {
        out.push(decimal);
        out.push_str(frac_part);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-04-30 00:00:00 UTC = day 20_573 since 1970-01-01.
    // 20_573 * 86_400 * 1_000 = 1_777_507_200_000 ms.
    const TS_2026_04_30: i64 = 1_777_507_200_000;

    #[test]
    fn format_date_english_medium() {
        assert_eq!(
            format_date(TS_2026_04_30, "en", FormatStyle::Medium),
            "30 Apr 2026"
        );
    }

    #[test]
    fn format_date_english_long() {
        assert_eq!(
            format_date(TS_2026_04_30, "en", FormatStyle::Long),
            "30 April 2026"
        );
    }

    #[test]
    fn format_date_italian_long() {
        assert_eq!(
            format_date(TS_2026_04_30, "it", FormatStyle::Long),
            "30 aprile 2026"
        );
    }

    #[test]
    fn format_date_unknown_locale_falls_back_to_english() {
        assert_eq!(
            format_date(TS_2026_04_30, "jp", FormatStyle::Long),
            "30 April 2026"
        );
    }

    #[test]
    fn format_number_english_thousands() {
        assert_eq!(format_number(1234.56, "en", FormatStyle::Medium), "1,234.56");
    }

    #[test]
    fn format_number_italian_thousands() {
        assert_eq!(format_number(1234.56, "it", FormatStyle::Medium), "1.234,56");
    }

    #[test]
    fn format_number_negative() {
        assert_eq!(format_number(-1234.56, "it", FormatStyle::Medium), "-1.234,56");
    }

    #[test]
    fn format_number_under_thousand() {
        assert_eq!(format_number(42.5, "en", FormatStyle::Medium), "42.50");
    }

    #[test]
    fn format_currency_usd_english() {
        assert_eq!(format_currency(1234.56, "USD", "en"), "$1,234.56");
    }

    #[test]
    fn format_currency_eur_italian() {
        assert_eq!(format_currency(1234.56, "EUR", "it"), "1.234,56 €");
    }

    #[test]
    fn format_currency_gbp_english() {
        assert_eq!(format_currency(50.0, "GBP", "en"), "£50.00");
    }

    #[test]
    fn format_currency_unknown_code_passes_through() {
        // No symbol entry → use the code verbatim.
        assert_eq!(format_currency(100.0, "XAU", "en"), "XAU100.00");
    }

    #[test]
    fn format_style_parse_falls_back_to_medium() {
        assert_eq!(FormatStyle::parse("nonsense"), FormatStyle::Medium);
        assert_eq!(FormatStyle::parse(""), FormatStyle::Medium);
        assert_eq!(FormatStyle::parse("long"), FormatStyle::Long);
    }
}
