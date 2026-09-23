use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

fn output_directory(label: &str) -> std::path::PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("week4-{label}-{suffix}"))
}

#[test]
fn field_writes_both_contract_cases() {
    let exact = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "8", "--nu", "0.1", "--t", "1"])
        .output()
        .unwrap();
    assert!(exact.status.success());
    let value: serde_json::Value = serde_json::from_slice(&exact.stdout).unwrap();
    assert_eq!(value["case"], "taylor-green");
    assert_eq!(value["u"].as_array().unwrap().len(), 64);

    let random = Command::new(env!("CARGO_BIN_EXE_field"))
        .args([
            "random", "--n", "32", "--seed", "2026", "--k-min", "2", "--k-max", "6",
        ])
        .output()
        .unwrap();
    assert!(random.status.success());
    let value: serde_json::Value = serde_json::from_slice(&random.stdout).unwrap();
    assert_eq!(value["seed"], 2026);
    assert_eq!(value["k_band"], serde_json::json!([2, 6]));
}

#[test]
fn fluid_writes_run_and_six_decimal_frames() {
    let initial = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "8"])
        .output()
        .unwrap();
    let output = output_directory("fluid");
    let mut child = Command::new(env!("CARGO_BIN_EXE_fluid"))
        .args([
            "--method", "rk4", "--nu", "0.1", "--dt", "0.01", "--t-end", "0.02", "--every", "0.01",
            "--out",
        ])
        .arg(&output)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&initial.stdout)
        .unwrap();
    let result = child.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["method"], "rk4");
    let frames = fs::read_to_string(output.join("fields.jsonl")).unwrap();
    assert_eq!(frames.lines().count(), 3);
    assert!(frames.contains("\"t\":0.010000"));
    fs::remove_dir_all(output).unwrap();
}
