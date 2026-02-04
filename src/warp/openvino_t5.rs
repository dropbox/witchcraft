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

use log::{debug, info};
use crate::embed_zst_asset;
use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use openvino::{Core, DeviceType, InferRequest, Shape};
use std::cell::RefCell;
use std::path::PathBuf;
use tokenizers::Tokenizer;

// Asset definitions - zstd-compressed model files
embed_zst_asset!(pub TOKENIZER, "tokenizer.json.zst");

// INT4 quantized model (98.9% accuracy, 6.38x compression)
embed_zst_asset!(pub MODEL_INT4_XML, "xtr-ov-int4.xml.zst");
embed_zst_asset!(pub MODEL_INT4_BIN, "xtr-ov-int4.bin.zst");

pub struct T5ModelBuilder {
}

impl T5ModelBuilder {
    /// Load the T5 model configuration and tokenizer from assets.
    ///
    /// This method loads the config.json and tokenizer.json files,
    /// which are shared across all T5 backend implementations.
    pub fn load(assets: &PathBuf) -> Result<(Self, Tokenizer)> {
        // Load tokenizer
        let tok_bytes = TOKENIZER
            .bytes(assets)
            .map_err(|_| anyhow!("failed to get decompressed bytes for TOKENIZER"))?;
        let tokenizer = Tokenizer::from_bytes(tok_bytes)
            .map_err(|e| anyhow!("failed to create tokenizer: {}", e))?;

        Ok((Self { }, tokenizer))
    }

    /// Build the T5 encoder model using OpenVINO with INT4 quantization.
    pub fn build_encoder(&self, device: &Device, assets: &PathBuf) -> Result<T5EncoderModel> {
        // Decompress model files from embedded assets
        let xml_bytes = MODEL_INT4_XML.bytes(assets)
            .map_err(|_| anyhow!("failed to get decompressed bytes for INT4 MODEL_XML"))?;
        let bin_bytes = MODEL_INT4_BIN.bytes(assets)
            .map_err(|_| anyhow!("failed to get decompressed bytes for INT4 MODEL_BIN"))?;

        // Create temporary directory for model files
        // OpenVINO's read_model API requires file paths
        let temp_dir = tempfile::tempdir()
            .map_err(|e| anyhow!("failed to create temp directory: {}", e))?;
        let xml_path = temp_dir.path().join("model-int4.xml");
        let bin_path = temp_dir.path().join("model-int4.bin");

        std::fs::write(&xml_path, xml_bytes)
            .map_err(|e| anyhow!("failed to write model.xml: {}", e))?;
        std::fs::write(&bin_path, bin_bytes)
            .map_err(|e| anyhow!("failed to write model.bin: {}", e))?;

        // Initialize OpenVINO Core
        let mut core = Core::new()
            .map_err(|e| anyhow!("failed to create OpenVINO Core: {:?}", e))?;

        // Read the model
        let model = core
            .read_model_from_file(
                xml_path.to_str().ok_or_else(|| anyhow!("invalid path"))?,
                bin_path.to_str().ok_or_else(|| anyhow!("invalid path"))?,
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
            core.set_property(&ov_device, &RwPropertyKey::HintInferencePrecision, &precision)
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

        // Temporary files will be automatically cleaned up when temp_dir is dropped

        Ok(T5EncoderModel {
            ov_infer_request: RefCell::new(infer_request),
            device: device.clone(),
        })
    }
}

/// T5 Encoder Model using OpenVINO backend.
///
/// This struct wraps an OpenVINO compiled model and inference request,
/// while maintaining a Candle Device for API compatibility.
/// Uses RefCell for interior mutability of the inference request.
pub struct T5EncoderModel {
    ov_infer_request: RefCell<InferRequest>,
    device: Device,
}

impl T5EncoderModel {
    /// Perform forward pass through the T5 encoder.
    ///
    /// This method converts the input Candle tensor to OpenVINO format,
    /// runs inference, and converts the output back to a Candle tensor.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs as a Candle tensor [batch_size, seq_len]
    ///
    /// # Returns
    /// Output embeddings as a Candle tensor [batch_size, seq_len, embedding_dim]
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Get input dimensions
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

        // Convert Candle tensor to Vec<i64>
        let input_data: Vec<u32> = input_ids
            .flatten_all()?
            .to_vec1::<u32>()
            .map_err(|e| anyhow!("failed to convert input tensor to vec: {}", e))?;

        let input_data: Vec<i64> = input_data.into_iter()
            .map(i64::from) // or .map(|x| x.into())
            .collect();

        // Create OpenVINO shape (OpenVINO uses i64 for dimensions)
        let ov_shape = Shape::new(&[batch_size as i64, seq_len as i64])
            .map_err(|e| anyhow!("failed to create OpenVINO shape: {:?}", e))?;

        // Create OpenVINO tensor
        let mut ov_input = openvino::Tensor::new(
            openvino::ElementType::I64,
            &ov_shape
        ).map_err(|e| anyhow!("failed to create OpenVINO input tensor: {:?}", e))?;

        // Set tensor data
        let ov_data = ov_input.get_data_mut::<i64>()
            .map_err(|e| anyhow!("failed to get mutable data: {:?}", e))?;

        debug!("[DEBUG] OpenVINO tensor buffer size: {}, input data size: {}", ov_data.len(), input_data.len());
        if ov_data.len() != input_data.len() {
            return Err(anyhow!("SIZE MISMATCH: OpenVINO buffer has {} elements but input has {} elements!",
                ov_data.len(), input_data.len()));
        }

        ov_data.copy_from_slice(&input_data);

        let mut infer_request = self.ov_infer_request.borrow_mut();

        // Set input tensor (no index parameter needed)
        infer_request
            .set_input_tensor(&ov_input)
            .map_err(|e| anyhow!("failed to set input tensor: {:?}", e))?;

        // Run inference
        infer_request
            .infer()
            .map_err(|e| anyhow!("OpenVINO inference failed: {:?}", e))?;

        // Get output tensor (no index parameter needed)
        let ov_output = infer_request
            .get_output_tensor()
            .map_err(|e| anyhow!("failed to get output tensor: {:?}", e))?;

        // Get output shape and data
        let output_shape = ov_output.get_shape()
            .map_err(|e| anyhow!("failed to get output shape: {:?}", e))?;

        // Convert Shape to Vec<usize> - OpenVINO uses i64 for dimensions
        let output_dims: Vec<usize> = output_shape
            .get_dimensions()
            .iter()
            .map(|&d| d as usize)
            .collect();

        // Copy data from OpenVINO tensor
        let output_data: Vec<f32> = ov_output.get_data::<f32>()
            .map_err(|e| anyhow!("failed to get output data: {:?}", e))?
            .to_vec();

        /*
        // Debug: Check for NaN values
        let has_nan = output_data.iter().any(|x| x.is_nan());
        let has_inf = output_data.iter().any(|x| x.is_infinite());
        if has_nan || has_inf {
            warn!("OpenVINO output contains NaN={} Inf={}", has_nan, has_inf);
            warn!("First 10 values: {:?}", &output_data[..10.min(output_data.len())]);
            warn!("Last 10 values: {:?}", &output_data[output_data.len().saturating_sub(10)..]);

            // Find which rows have NaN
            let embedding_dim = output_dims[2];
            let num_tokens = output_dims[1];
            let mut nan_rows: Vec<usize> = vec![];
            for row in 0..num_tokens {
                let row_start = row * embedding_dim;
                let row_data = &output_data[row_start..row_start + embedding_dim];
                if row_data.iter().any(|x| x.is_nan()) {
                    nan_rows.push(row);
                }
            }
            warn!("Rows with NaN (total {} out of {}): {:?}", nan_rows.len(), num_tokens, nan_rows);
            warn!("Input dimensions: batch={}, seq_len={}", batch_size, seq_len);
            warn!("Output dimensions: {:?}", output_dims);

            // Show input tokens at NaN positions
            warn!("Input token IDs (first 20): {:?}", &input_data[..20.min(input_data.len())]);
            warn!("Input token IDs (last 20): {:?}", &input_data[input_data.len().saturating_sub(20)..]);
            if nan_rows.len() <= 10 {
                warn!("Token IDs at NaN row positions:");
                for &row_idx in &nan_rows {
                    if row_idx < input_data.len() {
                        warn!("  Row {}: token_id={}", row_idx, input_data[row_idx]);
                    }
                }
            }

            // Check if last row has NaN
            if nan_rows.contains(&(num_tokens - 1)) {
                warn!("LAST ROW (index {}) HAS NaN!", num_tokens - 1);
                warn!("This suggests an off-by-one error or padding issue");
            }

            assert!(false);
        }
        */

        // Convert back to Candle tensor
        let output_tensor = Tensor::from_vec(output_data, &output_dims[..], &self.device)
            .map_err(|e| anyhow!("failed to create output Candle tensor: {}", e))?;

        //info!("got {}", output_tensor);
        Ok(output_tensor)
    }

    /// Returns the Candle device for API compatibility.
    pub fn device(&self) -> &Device {
        &self.device
    }
}
