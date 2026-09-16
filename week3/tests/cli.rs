use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn output_directory(label: &str) -> std::path::PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ising-{label}-{suffix}"))
}

#[test]
fn cli_writes_contract_files_and_restarts_measurement_sweep() {
    let output = output_directory("contract");
    let result = Command::new(env!("CARGO_BIN_EXE_ising"))
        .args([
            "--update",
            "metropolis",
            "--l",
            "4",
            "--t-from",
            "2.0",
            "--t-to",
            "2.1",
            "--t-step",
            "0.1",
            "--discard",
            "2",
            "--measure",
            "3",
            "--every",
            "2",
            "--seed",
            "2026",
            "--out",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let run = fs::read_to_string(output.join("run.json")).unwrap();
    assert!(run.contains("\"t_grid\":[2.000000,2.100000]"));
    assert!(run.contains("\"time_unit\":\"sweep\""));

    let series = fs::read_to_string(output.join("series.jsonl")).unwrap();
    assert_eq!(series.lines().count(), 6);
    assert_eq!(series.matches("\"sweep\":1").count(), 2);

    let frames = fs::read_to_string(output.join("spins.jsonl")).unwrap();
    assert_eq!(frames.lines().count(), 2);
    fs::remove_dir_all(output).unwrap();
}

#[test]
fn wolff_rows_include_cluster_size() {
    let output = output_directory("wolff");
    let result = Command::new(env!("CARGO_BIN_EXE_ising"))
        .args([
            "--update",
            "wolff",
            "--l",
            "4",
            "--t-from",
            "2.3",
            "--t-to",
            "2.3",
            "--t-step",
            "0.1",
            "--discard",
            "2",
            "--measure",
            "3",
            "--seed",
            "42",
            "--out",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let run = fs::read_to_string(output.join("run.json")).unwrap();
    assert!(run.contains("\"time_unit\":\"cluster_flip\""));
    let series = fs::read_to_string(output.join("series.jsonl")).unwrap();
    assert_eq!(series.matches("\"cluster_size\":").count(), 3);
    fs::remove_dir_all(output).unwrap();
}
