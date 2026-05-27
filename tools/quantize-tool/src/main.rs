use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Result};
use std::env;

fn run_quantize_safetensors(
    in_file: std::path::PathBuf,
    out_path: std::path::PathBuf,
) -> Result<()> {
    let tensors = candle_core::safetensors::load(in_file, &Device::Cpu)?;
    println!("tensors: {}", tensors.len());

    let qtensors = tensors
        .into_iter()
        .map(|(name, tensor)| {
            let qdtype = if tensor.rank() == 2 {
                let dim1 = tensor.dim(1)?;
                if dim1 % GgmlDType::Q4K.block_size() == 0 {
                    GgmlDType::Q4K
                } else if dim1 % GgmlDType::Q4_1.block_size() == 0 {
                    GgmlDType::Q4_1
                } else {
                    GgmlDType::F32
                }
            } else {
                GgmlDType::F32
            };
            println!("  {name} {qdtype:?} {tensor:?}");
            let tensor = QTensor::quantize(&tensor, qdtype)?;
            Ok((name, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    let qtensors = qtensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();

    let mut out = std::fs::File::create(out_path)?;
    gguf_file::write(&mut out, &[], &qtensors)?;
    out.sync_all()?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let in_file = &args[1];
    let out_file = &args[2];
    run_quantize_safetensors(in_file.into(), out_file.into()).unwrap();
    Ok(())
}
