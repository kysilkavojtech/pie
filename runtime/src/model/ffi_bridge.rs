//! IPC bridge for cross-process Python communication.
//!
//! This module provides an async wrapper for IPC-based communication with
//! Python worker processes in the symmetric worker architecture.

use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};

/// Async IPC client for cross-process communication.
///
/// Uses ipc-channel to communicate with Python processes in other PIDs.
#[derive(Clone)]
pub struct AsyncIpcClient {
    backend: std::sync::Arc<crate::model::ffi_ipc::FfiIpcBackend>,
}

impl AsyncIpcClient {
    /// Create a new IPC client from an FfiIpcBackend.
    pub fn new(backend: std::sync::Arc<crate::model::ffi_ipc::FfiIpcBackend>) -> Self {
        Self { backend }
    }
    
    /// Call a Python method asynchronously via IPC.
    pub async fn call<T, R>(&self, method: &str, args: &T) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        // D1 IPC sub-bucket profiling. Disabled (compiled out) unless the
        // ipc-profiling Cargo feature is on. The four checkpoints are:
        //   t0 -> t1: Rust msgpack serialize
        //   t1 -> t2: send into ipc-channel + Python work + recv response
        //             (the "wire + python" bucket; subtract Python's
        //             [PROFILING] total to get the pipe-transit residual)
        //   t2 -> t3: Rust msgpack deserialize
        // See benches/docs/next_benchmark_plan_2026_04_08.md (D1).
        #[cfg(feature = "ipc-profiling")]
        let _t0 = std::time::Instant::now();

        // Serialize arguments
        let payload = rmp_serde::to_vec_named(args)
            .map_err(|e| anyhow::anyhow!("Failed to serialize args: {}", e))?;

        #[cfg(feature = "ipc-profiling")]
        let _t1 = std::time::Instant::now();
        #[cfg(feature = "ipc-profiling")]
        let _payload_bytes = payload.len();

        // Send via IPC
        let response = self.backend.call(method, payload).await?;

        #[cfg(feature = "ipc-profiling")]
        let _t2 = std::time::Instant::now();
        #[cfg(feature = "ipc-profiling")]
        let _response_bytes = response.len();

        // Deserialize response
        let result = rmp_serde::from_slice(&response)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize response: {}", e));

        #[cfg(feature = "ipc-profiling")]
        {
            let _t3 = std::time::Instant::now();
            eprintln!(
                "[IPC-PROFILE] {{\"method\":\"{}\",\"req_bytes\":{},\"resp_bytes\":{},\
                 \"serialize_ms\":{:.3},\"wire_python_ms\":{:.3},\
                 \"deserialize_ms\":{:.3},\"total_ms\":{:.3}}}",
                method,
                _payload_bytes,
                _response_bytes,
                _t1.duration_since(_t0).as_micros() as f64 / 1000.0,
                _t2.duration_since(_t1).as_micros() as f64 / 1000.0,
                _t3.duration_since(_t2).as_micros() as f64 / 1000.0,
                _t3.duration_since(_t0).as_micros() as f64 / 1000.0,
            );
        }

        result
    }
    
    /// Fire-and-forget notification.
    pub async fn notify<T>(&self, method: &str, args: &T) -> Result<()>
    where
        T: Serialize,
    {
        let _: () = self.call(method, args).await?;
        Ok(())
    }
    
    /// Call with timeout.
    pub async fn call_with_timeout<T, R>(
        &self,
        method: &str,
        args: &T,
        timeout: std::time::Duration,
    ) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        tokio::time::timeout(timeout, self.call(method, args))
            .await
            .map_err(|_| anyhow::anyhow!("IPC call timed out"))?
    }
}
