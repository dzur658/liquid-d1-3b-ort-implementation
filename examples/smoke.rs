use anyhow::Result;
use d1_onnx_testing::D1;

fn main() -> Result<()> {
    let mut d1 = D1::new()?;
    let p = d1.yes_no(
        "The experiment was a complete success and the results were published. Unfortunately, the experiment failed to replicate.",
        "Did the experiment succeed?",
    )?;
    println!("P(claim) = {p}"); // tune your flag threshold against this, recall-first
    Ok(())
}