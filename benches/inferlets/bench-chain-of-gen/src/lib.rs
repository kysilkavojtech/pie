//! Benchmark inferlet: Chain-of-Generations
//!
//! Performs 3 sequential generation steps (draft → critique → revise) within
//! a single inferlet invocation. The KV cache persists across all steps,
//! avoiding redundant prefill — this is the "Chat API Tax" that external
//! orchestration pays but Pie does not.

use inferlet::stop_condition::{ends_with_any, max_len, StopCondition};
use inferlet::{Args, Result, Sampler};
use std::time::Instant;

const CRITIQUE_PROMPT: &str = "\
Now critically evaluate the response above. Identify any inaccuracies, \
missing information, or areas that could be clearer. Be specific and concise.";

const REVISE_PROMPT: &str = "\
Based on the critique above, write an improved and corrected version of the \
original response. Make it accurate, clear, and complete.";

fn make_sampler(temperature: f32) -> Sampler {
    if temperature == 0.0 {
        Sampler::greedy()
    } else {
        Sampler::top_p(temperature, 0.95)
    }
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_tokens_per_step: usize = args
        .value_from_str(["-n", "--max-tokens-per-step"])
        .unwrap_or(256);
    let system: String = args
        .value_from_str(["-s", "--system"])
        .unwrap_or_else(|_| "You are a helpful, respectful and honest assistant.".to_string());
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.0);

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();
    let mut ctx = model.create_context();

    ctx.fill_system(&system);

    // Step 1: Draft
    let step1_start = Instant::now();
    ctx.fill_user(&prompt);
    let stop = max_len(max_tokens_per_step).or(ends_with_any(eos_tokens.clone()));
    let _draft = ctx.generate(make_sampler(temperature), stop).await;
    let step1_ms = step1_start.elapsed().as_millis();
    println!("STEP1_MS:{}", step1_ms);

    // Step 2: Critique (KV cache from step 1 is still present)
    let step2_start = Instant::now();
    ctx.fill_user(CRITIQUE_PROMPT);
    let stop = max_len(max_tokens_per_step).or(ends_with_any(eos_tokens.clone()));
    let _critique = ctx.generate(make_sampler(temperature), stop).await;
    let step2_ms = step2_start.elapsed().as_millis();
    println!("STEP2_MS:{}", step2_ms);

    // Step 3: Revise (KV cache from steps 1+2 is still present)
    let step3_start = Instant::now();
    ctx.fill_user(REVISE_PROMPT);
    let stop = max_len(max_tokens_per_step).or(ends_with_any(eos_tokens.clone()));
    let revised = ctx.generate(make_sampler(temperature), stop).await;
    let step3_ms = step3_start.elapsed().as_millis();
    println!("STEP3_MS:{}", step3_ms);

    // Report total token count estimate for prefill accounting
    let total_tokens = ctx.get_token_ids().len();
    println!("TOTAL_TOKENS:{}", total_tokens);

    Ok(revised)
}
