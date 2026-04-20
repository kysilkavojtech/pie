//! Log-likelihood inferlet for lm-eval-harness integration.
//!
//! Computes the cumulative log probability of each continuation string given
//! a shared context. Used by PieLM to support loglikelihood-based benchmarks
//! (ARC, MMLU, etc.) through lm-evaluation-harness.
//!
//! Protocol:
//!   --context "raw prompt text"
//!   --continuations '["choice A", "choice B", ...]'
//!   → returns JSON: [{"logprob": -2.1, "is_greedy": true}, ...]

use inferlet::{Args, Result};

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let context: String = args.value_from_str(["-c", "--context"])?;
    let continuations_json: String = args.value_from_str("--continuations")?;

    let continuations: Vec<String> = serde_json::from_str(&continuations_json)
        .map_err(|e| inferlet::anyhow!("Failed to parse continuations JSON: {e}"))?;

    let model = inferlet::get_auto_model();
    let mut ctx = model.create_context();

    // Fill raw context — lm-eval pre-formats the prompt, no chat template needed
    ctx.fill(&context);
    ctx.flush().await;

    let mut results = Vec::new();

    for (ci, continuation) in continuations.iter().enumerate() {
        let mut fork = ctx.fork();
        let tokens = fork.tokenizer.tokenize(continuation);
        let mut logprob = 0.0f32;
        let mut all_greedy = true;

        eprintln!("[DEBUG] continuation {ci}: {continuation:?} -> {tokens:?}");

        for (ti, &token_id) in tokens.iter().enumerate() {
            let dist = fork.decode_step_dist().await;

            eprintln!("[DEBUG]   step {ti}: want token_id={token_id}, dist has {} entries, top5_ids={:?}, top5_probs={:?}",
                dist.ids.len(),
                &dist.ids[..dist.ids.len().min(5)],
                &dist.probs[..dist.probs.len().min(5)]);

            // Check if this token is the greedy (argmax) choice
            if !dist.ids.is_empty() && dist.ids[0] != token_id {
                all_greedy = false;
            }

            // Find the token's probability in the distribution
            if let Some(idx) = dist.ids.iter().position(|&id| id == token_id) {
                let prob = dist.probs[idx];
                eprintln!("[DEBUG]   found at idx={idx}, prob={prob}");
                if prob > 0.0 {
                    logprob += prob.ln();
                } else {
                    logprob = f32::NEG_INFINITY;
                    all_greedy = false;
                    break;
                }
            } else {
                // Token not in top-k distribution — negligible probability
                eprintln!("[DEBUG]   NOT FOUND in distribution — setting -inf");
                logprob = f32::NEG_INFINITY;
                all_greedy = false;
                break;
            }

            // Advance context with this token for the next step
            fork.fill_token(token_id);
        }

        results.push(serde_json::json!({
            "logprob": logprob,
            "is_greedy": all_greedy,
        }));
    }

    Ok(serde_json::to_string(&results)?)
}
