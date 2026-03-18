//! T5 model implementation using OpenVINO backend.
//!
//! This module provides a T5 encoder implementation that uses Intel OpenVINO
//! for inference. It maintains API compatibility with the Candle-based backends
//! while leveraging OpenVINO's optimized runtime.
//!
//! Key characteristics:
//! - OpenVINO IR format (.xml + .bin files)
//! - Hybrid tensor approach: Candle tensors for API, OpenVINO for inference
//! - Cross-platform support (Windows, macOS)
//! - Compatible with existing embedder interface

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use openvino::{CompiledModel, Core, DeviceType, InferRequest, Shape};
use std::cell::RefCell;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::embed_asset;
embed_asset!(pub TOKENIZER, "tokenizer.json");

pub struct T5ModelBuilder {}

impl T5ModelBuilder {
    /// Load the T5 model configuration and tokenizer from assets.
    ///
    /// This method loads the config.json and tokenizer.json files,
    /// which are shared across all T5 backend implementations.
    pub fn load(assets: &Path) -> Result<(Self, Tokenizer)> {
        // On Windows, add assets directory to PATH early if it contains OpenVINO DLLs
        // This must happen before any OpenVINO code loads
        #[cfg(all(target_os = "windows", feature = "t5-openvino"))]
        {
            let dll_file = assets.join("openvino_c.dll");
            if dll_file.exists() {
                // Get absolute path
                let abs_assets = assets
                    .canonicalize()
                    .unwrap_or_else(|_| assets.to_path_buf());
                if let Some(assets_str) = abs_assets.to_str() {
                    // Add to PATH environment variable at the front
                    if let Ok(current_path) = std::env::var("PATH") {
                        let new_path = format!("{};{}", assets_str, current_path);
                        unsafe {
                            std::env::set_var("PATH", new_path);
                        }
                        log::info!("Added assets directory to PATH for OpenVINO DLLs",);
                    }
                }
            }
        }

        // Load tokenizer
        let tok_bytes = TOKENIZER
            .bytes(assets)
            .map_err(|_| anyhow!("failed to get decompressed bytes for TOKENIZER"))?;
        let tokenizer = Tokenizer::from_bytes(tok_bytes)
            .map_err(|e| anyhow!("failed to create tokenizer: {}", e))?;

        Ok((Self {}, tokenizer))
    }

    /// Build the T5 encoder model using OpenVINO with INT4 quantization.
    pub fn build_encoder(&self, device: &Device, assets: &Path) -> Result<T5EncoderModel> {
        // Initialize OpenVINO Core
        let mut core =
            Core::new().map_err(|e| anyhow!("failed to create OpenVINO Core: {:?}", e))?;

        // Load model files directly from assets directory
        log::info!("loading OpenVINO model...");
        let xml_path = assets.join("xtr-ov-int4.xml");
        let bin_path = assets.join("xtr-ov-int4.bin");

        let model = core
            .read_model_from_file(
                xml_path
                    .to_str()
                    .ok_or_else(|| anyhow!("invalid XML path"))?,
                bin_path
                    .to_str()
                    .ok_or_else(|| anyhow!("invalid BIN path"))?,
            )
            .map_err(|e| anyhow!("failed to read OpenVINO model: {:?}", e))?;

        // Determine OpenVINO device
        // Default to CPU for INT4 models - GPU has NaN issues with real text inputs
        // While GPU benchmarks show 2x speedup on synthetic inputs, real text produces NaN
        let ov_device = DeviceType::CPU;
        // GPU INT4 models require FP32 inference precision to prevent NaN
        // GPU INT4 accumulates numerical errors across multiple inferences causing NaN
        // FP32 intermediate computations prevent this while maintaining INT4 compression
        // Root cause: GPU state corruption after ~9 sequences with default precision
        // Can be overridden via OPENVINO_PRECISION environment variable
        if matches!(ov_device, DeviceType::GPU) {
            use openvino::RwPropertyKey;
            let precision = "f32".to_string();
            core.set_property(
                &ov_device,
                &RwPropertyKey::HintInferencePrecision,
                &precision,
            )
            .map_err(|e| anyhow!("failed to set GPU precision hint: {:?}", e))?;
        }

        // Compile the model for the target device
        let mut compiled_model = core
            .compile_model(&model, ov_device)
            .map_err(|e| anyhow!("failed to compile OpenVINO model: {:?}", e))?;

        // Create an inference request
        let infer_request = compiled_model
            .create_infer_request()
            .map_err(|e| anyhow!("failed to create inference request: {:?}", e))?;

        Ok(T5EncoderModel {
            ov_infer_request: RefCell::new(infer_request),
            device: device.clone(),
            _compiled_model: compiled_model,
            _core: core,
        })
    }
}

/// Round up to the next power of 2, with a minimum of 64.
/// This limits the number of distinct shapes OpenVINO sees, preventing
/// a memory leak in its CPU plugin that occurs on every shape transition.
fn bucket_size(n: usize) -> usize {
    let min = 64;
    if n <= min {
        return min;
    }
    n.next_power_of_two()
}

/// T5 Encoder Model using OpenVINO backend.
///
/// This struct wraps an OpenVINO compiled model and inference request,
/// while maintaining a Candle Device for API compatibility.
/// Uses RefCell for interior mutability of the inference request.
/// OpenVINO internally memory-maps model files for efficient loading.
pub struct T5EncoderModel {
    ov_infer_request: RefCell<InferRequest>,
    device: Device,
    // Prevent premature destruction of the C++ objects backing the InferRequest.
    // Rust drops fields in declaration order, so IR drops before these.
    _compiled_model: CompiledModel,
    _core: Core,
}

impl T5EncoderModel {
    /// Perform forward pass through the T5 encoder.
    ///
    /// Pads input to a power-of-2 bucket size to work around an OpenVINO CPU
    /// plugin memory leak triggered by frequent input shape changes.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let shape = input_ids.shape();
        let dims = shape.dims();
        if dims.len() != 2 {
            return Err(anyhow!(
                "Expected 2D input tensor [batch_size, seq_len], got {:?}",
                dims
            ));
        }
        let batch_size = dims[0];
        let seq_len = dims[1];
        let padded_len = bucket_size(seq_len);

        let input_data: Vec<u32> = input_ids
            .flatten_all()?
            .to_vec1::<u32>()
            .map_err(|e| anyhow!("failed to convert input tensor to vec: {}", e))?;

        // Pad with 0 (T5 pad token) to the bucket size
        let mut padded: Vec<i64> = input_data.into_iter().map(i64::from).collect();
        padded.resize(batch_size * padded_len, 0);

        let ov_shape = Shape::new(&[batch_size as i64, padded_len as i64])
            .map_err(|e| anyhow!("failed to create OpenVINO shape: {:?}", e))?;

        let mut ov_input = openvino::Tensor::new(openvino::ElementType::I64, &ov_shape)
            .map_err(|e| anyhow!("failed to create OpenVINO input tensor: {:?}", e))?;

        ov_input
            .get_data_mut::<i64>()
            .map_err(|e| anyhow!("failed to get mutable data: {:?}", e))?
            .copy_from_slice(&padded);

        let mut infer_request = self.ov_infer_request.borrow_mut();

        infer_request
            .set_input_tensor(&ov_input)
            .map_err(|e| anyhow!("failed to set input tensor: {:?}", e))?;

        infer_request
            .infer()
            .map_err(|e| anyhow!("OpenVINO inference failed: {:?}", e))?;

        let ov_output = infer_request
            .get_output_tensor()
            .map_err(|e| anyhow!("failed to get output tensor: {:?}", e))?;

        let output_shape = ov_output
            .get_shape()
            .map_err(|e| anyhow!("failed to get output shape: {:?}", e))?;

        let output_dims: Vec<usize> = output_shape
            .get_dimensions()
            .iter()
            .map(|&d| d as usize)
            .collect();

        let embedding_dim = *output_dims
            .last()
            .ok_or_else(|| anyhow!("empty output dimensions"))?;

        let output_data: Vec<f32> = ov_output
            .get_data::<f32>()
            .map_err(|e| anyhow!("failed to get output data: {:?}", e))?
            .to_vec();

        // Prevent ov_tensor_free on the IR's internal output buffer.
        // See OPENVINO_INT4_CRASHES.md "The mem::forget wrappers" section.
        std::mem::forget(ov_output);

        // Slice off padding: keep only the first seq_len token embeddings
        let unpadded: Vec<f32> = output_data
            .chunks(padded_len * embedding_dim)
            .flat_map(|batch| batch[..seq_len * embedding_dim].iter().copied())
            .collect();

        let output_tensor = Tensor::from_vec(
            unpadded,
            &[batch_size, seq_len, embedding_dim],
            &self.device,
        )
        .map_err(|e| anyhow!("failed to create output Candle tensor: {}", e))?;

        Ok(output_tensor)
    }

    /// Returns the Candle device for API compatibility.
    pub fn device(&self) -> &Device {
        &self.device
    }
}
