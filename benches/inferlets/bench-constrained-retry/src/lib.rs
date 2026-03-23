//! Benchmark inferlet: Constrained Retry
//!
//! Generates JSON output, validates it, and on failure restores the KV cache
//! to the pre-generation checkpoint to retry without re-prefilling the prompt.
//!
//! This demonstrates Pie's ability to "rewind" generation to a branch point,
//! something impossible with standard chat APIs where a failed generation
//! means re-sending the entire context.

use inferlet::stop_condition::{ends_with_any, max_len, StopCondition};
use inferlet::{Args, Result, Sampler};
use serde_json;
use std::time::Instant;

const DEFAULT_SYSTEM: &str = "\
You are a JSON assistant. You MUST respond with valid JSON only. \
No markdown, no explanation, no text outside the JSON object.";

fn make_sampler(temperature: f32) -> Sampler {
    Sampler::top_p(temperature, 0.95)
}

/// Simple JSON validation: check that the output parses as valid JSON.
fn is_valid_json(s: &str) -> bool {
    let trimmed = s.trim();
    if let Some(start) = trimmed.find('{') {
        let candidate = &trimmed[start..];
        serde_json::from_str::<serde_json::Value>(candidate).is_ok()
    } else if let Some(start) = trimmed.find('[') {
        let candidate = &trimmed[start..];
        serde_json::from_str::<serde_json::Value>(candidate).is_ok()
    } else {
        serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
    }
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(512);
    let max_retries: usize = args.value_from_str(["-r", "--max-retries"]).unwrap_or(5);
    let system: String = args
        .value_from_str(["-s", "--system"])
        .unwrap_or_else(|_| DEFAULT_SYSTEM.to_string());
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.6);

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();

    // Build the prompt context and flush to compute KV cache
    let start = Instant::now();
    let mut ctx = model.create_context();
    ctx.fill_system(&system);
    ctx.fill_user(&prompt);
    ctx.flush().await;

    let prefill_ms = start.elapsed().as_millis();
    println!("PREFILL_MS:{}", prefill_ms);

    let prefix_tokens = ctx.get_token_ids().len();
    println!("PREFIX_TOKENS:{}", prefix_tokens);

    // checkpoint is the flushed context — fork() gives copy-on-write KV cache
    let checkpoint = ctx;

    let mut best_output = String::new();

    for attempt in 0..=max_retries {
        let attempt_start = Instant::now();

        // Fork from checkpoint — shares KV cache pages copy-on-write,
        // no re-prefill needed. fork() handles the seed-token invariant.
        let mut attempt_ctx = checkpoint.fork();

        let stop = max_len(max_tokens).or(ends_with_any(eos_tokens.clone()));
        let output = attempt_ctx.generate(make_sampler(temperature), stop).await;
        let attempt_ms = attempt_start.elapsed().as_millis();
        let gen_tokens = attempt_ctx.get_token_ids().len() - prefix_tokens;

        println!("ATTEMPT_{}_MS:{}", attempt, attempt_ms);
        println!("ATTEMPT_{}_GEN_TOKENS:{}", attempt, gen_tokens);

        if is_valid_json(&output) {
            println!("ATTEMPT_{}_VALID:true", attempt);
            println!("TOTAL_ATTEMPTS:{}", attempt + 1);
            println!("TOTAL_PREFILL_TOKENS:{}", prefix_tokens);
            best_output = output;
            break;
        } else {
            println!("ATTEMPT_{}_VALID:false", attempt);
            if attempt == max_retries {
                println!("TOTAL_ATTEMPTS:{}", attempt + 1);
                println!("TOTAL_PREFILL_TOKENS:{}", prefix_tokens);
                best_output = output; // Return last attempt even if invalid
            }
        }
    }

    Ok(best_output)
}
