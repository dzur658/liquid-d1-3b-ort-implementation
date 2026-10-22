use anyhow::Result;
use d1_yesno::D1;

fn main() -> Result<()> {
    let mut d1 = D1::new()?;
    let p = d1.yes_no(
        "The experiment was a complete success and the results were published. Unfortunately, the experiment failed to replicate.",
        "Did the experiment succeed?",
    )?;
    println!("P(claim) = {p}");

    let p2 = d1.yes_no(
        "There is an inbound malicious email.",
        "Should the email be forwarded to the security team?",
    )?;
    println!("P(claim) = {p2}");

    let p3 = d1.yes_no(
        "A boy holds an ice cream cone in his hand. It is August in Alabama.",
        "If the boy does not eat the ice cream cone for 3 hours, will the ice cream melt?",
    )?;
    println!("P(claim) = {p3}");

    Ok(())
}