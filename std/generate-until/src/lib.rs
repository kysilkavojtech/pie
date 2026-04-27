//! Generate-until inferlet for lm-eval-harness integration.
//!
//! Generates text from a raw (pre-formatted) prompt, stopping at EOS tokens
//! or any of the caller-provided stop strings. Unlike text-completion, this
//! inferlet uses raw prompt filling (no chat template) and tokenizes stop
//! strings into token-level stop conditions so generation halts immediately.
//!
//! Protocol:
//!   --prompt "raw prompt text"
//!   --max-tokens 8192
//!   --temperature 0.0
//!   --stop '["Problem:", "\n\n"]'
//!   → returns generated text (stop string excluded)

use inferlet::stop_condition::{ends_with_any, max_len};
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

    // Build stop conditions: EOS tokens + caller-provided stop strings
    let mut stop_token_seqs = model.eos_tokens();
    for s in &stop_strings {
        let token_ids = tokenizer.tokenize(s);
        if !token_ids.is_empty() {
            stop_token_seqs.push(token_ids);
        }
    }

    let sampler = if temperature == 0.0 {
        Sampler::greedy()
    } else {
        Sampler::top_p(temperature, top_p)
    };
    let stop_cond = max_len(max_num_outputs).or(ends_with_any(stop_token_seqs));

    let mut output = ctx.generate(sampler, stop_cond).await;

    // Trim the stop string from the output if present
    for s in &stop_strings {
        if output.ends_with(s) {
            output.truncate(output.len() - s.len());
            break;
        }
    }

    Ok(output)
}
