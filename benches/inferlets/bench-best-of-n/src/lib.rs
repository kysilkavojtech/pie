//! Benchmark inferlet: Best-of-N
//!
//! Fills a shared prefix (system + user prompt) once, forks N contexts,
//! generates N candidates in parallel sharing the prefix KV cache, and
//! returns all outputs separated by a delimiter for external scoring.
//!
//! This avoids prefilling the prompt N times — the key advantage over
//! sending N independent API requests.

use futures::future;
use inferlet::stop_condition::{ends_with_any, max_len, StopCondition};
use inferlet::{Args, Result, Sampler};
use std::time::Instant;

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);
    let num_candidates: usize = args
        .value_from_str(["-c", "--num-candidates"])
        .unwrap_or(4);
    let system: String = args
        .value_from_str(["-s", "--system"])
        .unwrap_or_else(|_| "You are a helpful, respectful and honest assistant.".to_string());
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.6);

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();

    // Build shared prefix context and flush to compute KV cache once
    let start = Instant::now();
    let mut common = model.create_context();
    common.fill_system(&system);
    common.fill_user(&prompt);
    common.flush().await;
    let prefill_ms = start.elapsed().as_millis();
    println!("PREFILL_MS:{}", prefill_ms);

    let prefix_tokens = common.get_token_ids().len();
    println!("PREFIX_TOKENS:{}", prefix_tokens);

    // Fork N contexts and generate in parallel
    let gen_start = Instant::now();
    let sampler = Sampler::top_p(temperature, 0.95);

    let handles: Vec<_> = (0..num_candidates)
        .map(|i| {
            let mut ctx = common.fork();
            let eos = eos_tokens.clone();
            let sampler = sampler.clone();
            async move {
                let stop = max_len(max_tokens).or(ends_with_any(eos));
                let output = ctx.generate(sampler, stop).await;
                let tokens = ctx.get_token_ids().len();
                (i, output, tokens)
            }
        })
        .collect();

    let results = future::join_all(handles).await;
    let gen_ms = gen_start.elapsed().as_millis();
    println!("GENERATION_MS:{}", gen_ms);

    // Report per-candidate token counts and collect outputs
    let mut outputs = Vec::with_capacity(num_candidates);
    let mut total_gen_tokens = 0usize;
    for (i, output, tokens) in &results {
        let gen_tokens = tokens - prefix_tokens;
        total_gen_tokens += gen_tokens;
        println!("CANDIDATE_{}_TOKENS:{}", i, gen_tokens);
        outputs.push(output.as_str());
    }
    println!("TOTAL_GEN_TOKENS:{}", total_gen_tokens);
    // Prefill was done once; without sharing it would be N times
    println!("TOTAL_PREFILL_TOKENS:{}", prefix_tokens);

    // Return all candidates separated by delimiter for harness to score
    Ok(outputs.join("\n---CANDIDATE_SEPARATOR---\n"))
}
