//! Benchmark inferlet: Overhead measurement
//!
//! Three modes to isolate different layers of overhead:
//!
//! - `noop`: Returns immediately. Measures WASM instantiation + IPC round-trip.
//! - `flush-only`: Fills a single token and flushes. Measures instantiation + one
//!   prefill round-trip to the GPU worker.
//! - `one-token`: Fills a token, flushes, generates one token. Measures instantiation +
//!   prefill + one decode step.

use inferlet::stop_condition::max_len;
use inferlet::{Args, Result, Sampler};
use std::time::Instant;

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let mode: String = args
        .value_from_str(["-m", "--mode"])
        .unwrap_or_else(|_| "noop".to_string());

    let t0 = Instant::now();

    match mode.as_str() {
        "noop" => {
            println!("MODE:noop");
            println!("WALL_MS:{}", t0.elapsed().as_micros() as f64 / 1000.0);
            Ok("noop".to_string())
        }
        "flush-only" => {
            println!("MODE:flush-only");
            let model = inferlet::get_auto_model();
            let t_model = Instant::now();
            println!("GET_MODEL_MS:{}", t_model.duration_since(t0).as_micros() as f64 / 1000.0);

            let mut ctx = model.create_context();
            ctx.fill_user("x");
            let t_fill = Instant::now();
            println!("FILL_MS:{}", t_fill.duration_since(t_model).as_micros() as f64 / 1000.0);

            ctx.flush().await;
            let t_flush = Instant::now();
            println!("FLUSH_MS:{}", t_flush.duration_since(t_fill).as_micros() as f64 / 1000.0);
            println!("WALL_MS:{}", t0.elapsed().as_micros() as f64 / 1000.0);

            Ok("flushed".to_string())
        }
        "one-token" => {
            println!("MODE:one-token");
            let model = inferlet::get_auto_model();
            let t_model = Instant::now();
            println!("GET_MODEL_MS:{}", t_model.duration_since(t0).as_micros() as f64 / 1000.0);

            let mut ctx = model.create_context();
            ctx.fill_user("Hello");
            ctx.flush().await;
            let t_flush = Instant::now();
            println!("PREFILL_MS:{}", t_flush.duration_since(t_model).as_micros() as f64 / 1000.0);

            let output = ctx.generate(Sampler::greedy(), max_len(1)).await;
            let t_gen = Instant::now();
            println!("DECODE_MS:{}", t_gen.duration_since(t_flush).as_micros() as f64 / 1000.0);
            println!("WALL_MS:{}", t0.elapsed().as_micros() as f64 / 1000.0);

            Ok(output)
        }
        other => {
            Ok(format!("unknown mode: {}", other))
        }
    }
}
