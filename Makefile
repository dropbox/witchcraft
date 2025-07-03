download:
	python downloadweights.py

build:
	cargo build --release --features accelerate
	ln -vf target/release/libwarp.dylib target/release/warp.node

buildemb:
	cargo build --release --features accelerate,embed-assets
	ln -vf target/release/libwarp.dylib target/release/warp.node

run: build
	node index.js
