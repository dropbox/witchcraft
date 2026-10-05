#!/usr/bin/env node
const path = require('path');
const util = require('util');
const exec = util.promisify(require('child_process').exec);

async function run(command) {
  console.log(`running: ${command} ...`);
  const { stdout, stderr } = await exec(command);
  if (stdout) console.log(stdout);
  if (stderr) console.error(stderr);
}

async function build(platform, arch) {
  if (platform == 'darwin') {
    await run(`cargo build --release --target aarch64-apple-darwin --features t5-quantized,metal,napi`);
    await run(`cargo build --release --target x86_64-apple-darwin --features t5-quantized,fbgemm,hybrid-dequant,napi`);
    await run(`lipo -create target/aarch64-apple-darwin/release/libwitchcraft.dylib target/x86_64-apple-darwin/release/libwitchcraft.dylib -output warp.node`);
  } else if (platform == 'win32') {
    var target = "";
    if (arch == "x64") {
      target = "x86_64";
    } else if (arch == "arm64") {
      target = "aarch64";
    } else {
      console.error(`unsupported target ${platform} ${arch}`);
      return;
    }
    await run(`cargo build --release --target ${target}-pc-windows-msvc --features modernbert-quantized,neso-d3d12,napi`);
    await run(`cp target/${target}-pc-windows-msvc/release/witchcraft.dll warp.node`);
  }
}

const platform = process.env.npm_config_platform || process.platform;
const arch = process.env.npm_config_arch || process.arch;

if (process.env.ENABLE_WARP == '1') {
  console.log(`building witchcraft native module for ${platform} ${arch}`);
  build(platform, arch);
} else {
  console.log(`Not building native module, set ENABLE_WARP=1 in environment to enable`);
}
