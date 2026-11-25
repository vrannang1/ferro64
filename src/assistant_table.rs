//! Tables a job saves as CSV: "every day, 10 doctors in Kolkata with name,
//! specialty, clinic, phone and source". The model ends its report with a
//! ```csv block; Ferro64 reads it, drops rows an earlier run already saved
//! and writes the files. The model never writes files itself.
//!
//! Rows are matched on their first two columns (name and clinic, say),
//! ignoring case, spacing and punctuation, so "Dr. A. Sen" at "Apollo" and
//! "dr a sen" at "APOLLO" are the same row.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const MAX_COLUMNS: usize = 12;
const MAX_ROWS: usize = 200;
const MAX_FIELD: usize = 500;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Table {
    pub columns: Vec<String>,
    /// Rows new in this run, which were saved.
    pub rows: Vec<Vec<String>>,
    /// Rows the model gave that an earlier run already had.
    pub repeats: Vec<Vec<String>>,
}

/// "name, specialty, clinic" → the column names, or nothing for a job with
/// no table.
pub fn parse_columns(spec: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    spec.split([',', '\n'])
        .map(|c| c.trim().chars().take(40).collect::<String>())
        .filter(|c| !c.is_empty() && seen.insert(normal(c)))
        .take(MAX_COLUMNS)
        .collect()
}

/// Lowercase letters and digits only, for matching.
fn normal(text: &str) -> String {
    text.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// What identifies a row: its first two columns, normalized.
pub fn key(row: &[String]) -> String {
    let parts: Vec<String> = row.iter().take(2).map(|f| normal(f)).collect();
    parts.join("|")
}

/// Takes the last ```csv block off a report and reads it into rows in the
/// job's column order. The header may name the columns in any order; one
/// that names none of them is read as data, in order.
pub fn extract(report: &str, columns: &[String]) -> (String, Option<Vec<Vec<String>>>) {
    let Some((start, end, body)) = last_csv_block(report) else {
        return (report.trim().to_string(), None);
    };
    let text = format!("{}{}", &report[..start], &report[end..]).trim().to_string();
    let mut reader = csv::ReaderBuilder::new().has_headers(false).flexible(true).trim(csv::Trim::All).from_reader(body.as_bytes());
    let mut records: Vec<Vec<String>> = reader
        .records()
        .filter_map(Result::ok)
        .map(|r| r.iter().map(|f| f.chars().take(MAX_FIELD).collect::<String>()).collect::<Vec<_>>())
        .filter(|r: &Vec<String>| r.iter().any(|f| !f.is_empty()))
        .collect();
    if records.is_empty() {
        return (text, Some(vec![]));
    }
    let wanted: Vec<String> = columns.iter().map(|c| normal(c)).collect();
    let header: Vec<String> = records[0].iter().map(|h| normal(h)).collect();
    // Where each of the job's columns sits in the model's rows.
    let positions: Vec<Option<usize>> = if header.iter().any(|h| wanted.contains(h)) {
        records.remove(0);
        wanted.iter().map(|w| header.iter().position(|h| h == w)).collect()
    } else {
        (0..columns.len()).map(Some).collect()
    };
    let rows = records
        .into_iter()
        .map(|r| positions.iter().map(|p| p.and_then(|i| r.get(i)).cloned().unwrap_or_default()).collect::<Vec<_>>())
        .filter(|r| !key(r).replace('|', "").is_empty())
        .take(MAX_ROWS)
        .collect();
    (text, Some(rows))
}

fn last_csv_block(report: &str) -> Option<(usize, usize, &str)> {
    let start = report.rfind("```csv").or_else(|| report.rfind("```CSV"))?;
    let body_start = report[start..].find('\n').map(|i| start + i + 1)?;
    let close = report[body_start..].find("```").map(|i| body_start + i);
    let body_end = close.unwrap_or(report.len());
    let end = close.map_or(report.len(), |c| c + 3);
    Some((start, end, &report[body_start..body_end]))
}

/// Splits rows into new ones and repeats, against keys already saved and
/// against each other.
pub fn split_new(rows: Vec<Vec<String>>, known: &HashSet<String>) -> (Vec<Vec<String>>, Vec<Vec<String>>) {
    let mut seen = known.clone();
    let (mut fresh, mut repeats) = (vec![], vec![]);
    for row in rows {
        if seen.insert(key(&row)) {
            fresh.push(row);
        } else {
            repeats.push(row);
        }
    }
    (fresh, repeats)
}

/// CSV text for download. A field from the web that a spreadsheet would run
/// as a formula (=, @, or + and - not followed by a phone number) gets a
/// leading apostrophe, so it shows as text.
pub fn to_csv(columns: &[String], rows: &[Vec<String>]) -> String {
    let mut writer = csv::Writer::from_writer(vec![]);
    let _ = writer.write_record(columns);
    for row in rows {
        let safe: Vec<String> = (0..columns.len()).map(|i| defuse(row.get(i).map_or("", String::as_str))).collect();
        let _ = writer.write_record(&safe);
    }
    String::from_utf8(writer.into_inner().unwrap_or_default()).unwrap_or_default()
}

fn defuse(field: &str) -> String {
    let phone_like = |rest: &str| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit() || " -()./".contains(c));
    let risky = match field.chars().next() {
        Some('=' | '@' | '\t' | '\r') => true,
        Some('+' | '-') => !phone_like(&field[1..]),
        _ => false,
    };
    if risky { format!("'{field}") } else { field.to_string() }
}

/// A folder name for a job.
pub fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-').chars().take(60).collect::<String>();
    if out.is_empty() { "job".into() } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cols() -> Vec<String> {
        parse_columns("name, clinic, city, phone, source")
    }

    #[test]
    fn columns_are_cleaned() {
        assert_eq!(parse_columns(" name ,clinic,, Name ,city"), ["name", "clinic", "city"]);
        assert!(parse_columns("  ").is_empty());
    }

    #[test]
    fn the_block_comes_off_and_maps_to_the_job_columns() {
        let report = "## Today\nFound 2 [S1].\n\n```csv\nClinic,Name,Phone,City\n\"Apollo, Salt Lake\",Dr. A. Sen,+91 33 1234 5678,Kolkata\nYashoda,Dr R Rao,,Hyderabad\n,,,\n```\nThanks";
        let (text, rows) = extract(report, &cols());
        assert_eq!(text, "## Today\nFound 2 [S1].\n\n\nThanks");
        let rows = rows.unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], ["Dr. A. Sen", "Apollo, Salt Lake", "Kolkata", "+91 33 1234 5678", ""]);
        // No header: read in order.
        let (_, rows) = extract("```csv\nDr X,Care,Hyderabad\n```", &cols());
        assert_eq!(rows.unwrap()[0][..3], ["Dr X", "Care", "Hyderabad"]);
        assert_eq!(extract("no table", &cols()).1, None);
    }

    #[test]
    fn repeats_match_loosely_and_within_a_run() {
        let row = |n: &str, c: &str| vec![n.to_string(), c.to_string()];
        let known: HashSet<String> = [key(&row("Dr. A. Sen", "Apollo"))].into();
        let (fresh, repeats) = split_new(vec![row("dr a sen", "APOLLO "), row("Dr B", "Care"), row("Dr. B", "care")], &known);
        assert_eq!(fresh, [row("Dr B", "Care")]);
        assert_eq!(repeats.len(), 2);
    }

    #[test]
    fn exported_fields_cannot_run_as_formulas() {
        let csv = to_csv(&cols(), &[vec!["=HYPERLINK(\"x\")".into(), "-cmd".into(), "Kolkata".into(), "+91 33 1234-5678".into()]]);
        assert!(csv.starts_with("name,clinic,city,phone,source\n"));
        assert!(csv.contains("'=HYPERLINK") && csv.contains("'-cmd") && csv.contains(",+91 33 1234-5678,"), "{csv}");
    }

    #[test]
    fn slugs_are_safe_folder_names() {
        assert_eq!(slug("Doctors: Kolkata / Hyderabad!"), "doctors-kolkata-hyderabad");
        assert_eq!(slug("../.."), "job");
    }
}
