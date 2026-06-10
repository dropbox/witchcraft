use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Result};
use std::env;
use std::path::PathBuf;

fn run_quantize_safetensors(in_file: PathBuf, out_path: PathBuf) -> Result<()> {
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
            let tensor = QTensor::quantize(&tensor, qdtype)?;
            Ok((name, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    let qtensors = qtensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect::<Vec<_>>();

    let mut out = std::fs::File::create(&out_path)?;
    gguf_file::write(&mut out, &[], &qtensors)?;
    out.sync_all()?;
    println!("wrote {}", out_path.display());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        anyhow::bail!("usage: {} IN.safetensors OUT.gguf", args[0]);
    }
    run_quantize_safetensors(PathBuf::from(&args[1]), PathBuf::from(&args[2]))?;
    Ok(())
}
