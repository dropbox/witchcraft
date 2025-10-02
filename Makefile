env/bin/activate:
	python3 -m venv env

env/bin/transformers-cli: env/bin/activate
	(source env/bin/activate; pip install -r requirements.txt)

assets:
	mkdir -p assets

assets/config.json.zst assets/tokenizer.json.zst xtr.safetensors assets/xtr.safetensors.zst: env/bin/transformers-cli
	(source env/bin/activate; python downloadweights.py)

assets/xtr.gguf.zst: xtr.safetensors
	cargo run --release --bin quantize-tool xtr.safetensors assets/xtr.gguf.zst

download: assets/config.json.zst assets/tokenizer.json.zst assets/xtr.safetensors.zst assets/xtr.gguf.zst

build: download
	RUSTFLAGS='-C target-feature=+neon' cargo build --release --target aarch64-apple-darwin --features accelerate
	ln -vf target/aarch64-apple-darwin/release/libwarp.dylib target/release/warp.node

buildemb: download
	cargo build --release --target aarch64-apple-darwin --features accelerate,embed-assets
	cargo build --release --target x86_64-apple-darwin --features accelerate,embed-assets
	ln -vf target/aarch64-apple-darwin/release/libwarp.dylib target/release/warp.node

module:
	cargo build --release --target aarch64-apple-darwin --features accelerate
	cargo build --release --target x86_64-apple-darwin --features accelerate
	lipo -create target/aarch64-apple-darwin/release/libwarp.dylib target/x86_64-apple-darwin/release/libwarp.dylib -output target/release/warp-macos-universal.node

winmodule:
	cargo xwin build --release --target x86_64-pc-windows-msvc
	ln -vf target/x86_64-pc-windows-msvc/release/warp.dll target/release/warp-windows.node

win: download
	RUSTFLAGS='-C target-feature=+avx2' cargo xwin build --release --target x86_64-pc-windows-msvc --features embed-assets

run: build
	node index.js

mcp: buildemb
	yarn napi build --release --features metal
	yarn tsc
	cmcp "node dist/index.js" tools/call name=search 'arguments:={"q": "teenagers and acne" }'
