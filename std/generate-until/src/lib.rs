//! Generate-until inferlet for lm-eval-harness integration.
//!
//! Generates text from a raw (pre-formatted) prompt, stopping at EOS tokens
//! or any of the caller-provided stop strings. Unlike text-completion, this
//! inferlet uses raw prompt filling (no chat template) and checks stop strings
//! against decoded text (not token IDs) to avoid tokenizer boundary mismatches.
//!
//! Protocol:
//!   --prompt "raw prompt text"
//!   --max-tokens 8192
//!   --temperature 0.0
//!   --stop '["Problem:", "\n\n"]'
//!   → returns generated text (stop string excluded)

use inferlet::stop_condition::StopCondition;
use inferlet::{Args, Result, Sampler};

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.0);
    let top_p: f32 = args.value_from_str("--top-p").unwrap_or(1.0);
    let stop_json: String = args
        .value_from_str("--stop")
        .unwrap_or_else(|_| "[]".to_string());

    let stop_strings: Vec<String> = serde_json::from_str(&stop_json)
        .map_err(|e| inferlet::anyhow!("Failed to parse stop JSON: {e}"))?;

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();
    let mut ctx = model.create_context();

    // Fill raw prompt — lm-eval pre-formats the prompt, no chat template needed
    ctx.fill(&prompt);

    let sampler = if temperature == 0.0 {
        Sampler::greedy()
    } else {
        Sampler::top_p(temperature, top_p)
    };

    // Build EOS token stop condition
    let eos_stop = inferlet::stop_condition::ends_with_any(model.eos_tokens());

    // Manual decode loop: check stop strings on decoded text to avoid
    // tokenizer boundary mismatches (e.g. "Problem:" tokenized differently
    // as a standalone string vs mid-generation after a newline).
    let mut generated_token_ids = Vec::new();

    loop {
        let next_token_id = ctx.decode_step(&sampler).await;
        ctx.fill_token(next_token_id);
        generated_token_ids.push(next_token_id);

        // Check EOS tokens (token-level is fine for these)
        if eos_stop.check(&generated_token_ids) {
            break;
        }

        // Check max length
        if generated_token_ids.len() >= max_num_outputs {
            break;
        }

        // Check stop strings on decoded text.
        // We check `contains` on the tail rather than `ends_with` because BPE
        // may merge the stop string with subsequent characters into one token
        // (e.g. "Problem:\n" as a single token), so ends_with("Problem:") would miss it.
        if !stop_strings.is_empty() {
            let text = tokenizer.detokenize(&generated_token_ids);
            let mut found_stop = false;
            for s in &stop_strings {
                if text.contains(s.as_str()) {
                    found_stop = true;
                    break;
                }
            }
            if found_stop {
                break;
            }
        }
    }

    let mut output = tokenizer.detokenize(&generated_token_ids);

    // Trim everything from the first occurrence of any stop string.
    // Because BPE can merge stop string + trailing chars into one token,
    // the stop string may not be at the very end of the output.
    let mut earliest_pos = output.len();
    for s in &stop_strings {
        if let Some(pos) = output.find(s.as_str()) {
            if pos < earliest_pos {
                earliest_pos = pos;
            }
        }
    }
    output.truncate(earliest_pos);

    Ok(output)
}
