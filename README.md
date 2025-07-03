# Rust-Warp #

This is a from-scratch reimplementation of Stanford's XTR-Warp
semantic search engine ( https://github.com/jlscheerer/xtr-warp ).

To run, you will need the XTR weights released by Google Deepmind, and
this repo contains python scripts for downloading them from Huggingface
and exporting to a single safetensors file:

* requirements.txt is for use with pip, to set up an env with
  the required python packages for the download scripts

* downloadxtr.py downloads the full repo, we only need the tokenizers

* downloadweights.py downloads the weights (again), adds the extra
  XTRLinear layer, and exports into xtr.safetensors. It also compressed
  the weights with zstd, unfortunately takes some time, but you only have
  to do it once.

(If you encounter TLS cerficiate errors running these scripts locally there is a
problem with your ZScaler cert setup.)

## Creating an index: ##

For testing, I have used the BEIR download script from XTR-Warp to download
nfcorpus, a dataset of ~3600 medical abstracts, and used the createdocs.py
script to dump them as individual files in the "documents" folder, which
you have to create yourself. Once that is present and populated, you can
run:

For your convenience, nfcorpus.tsv is included, so you can just do this:

```
$ cargo run --release readcsv datasets/nfcorpus.tsv
```

With all the documents in place, we can now create embeddings for them, with:

```
$ cargo run --release embed
```

This will look for documents that lack embeddings, and create and insert them.
The embeddings are stored in float-32 precision, so this will inflate the database
quite a bit. Technically we could drop the embeddings once they had been indexed,
but right now we prefer to keep them around to not have to recompute them, at
the expense of bloating up the database.

Next we create the index over the embeddings, with:

```
$ cargo run --release index
```

Which will scan the documents folder and create embedddings for everything. On
macOS, you can make it run faster with:

```
$ cargo run --release --features accelerate index
```

This will use hardware acceleration for the embeddings as well as for other
matrix ops.

All state gets persisted in mydb.sqlite, and you can abort the indexer and
it will pick up where it left off. To start over, you can just delete
mydb.sqlite and get a blank slate. Sqlite is just a practical way of gathering
everything, it is not involved in the actual index search.

## Querying the index ##

When you have the index, you can query it with:

```
$ cargo run --release query "does milk intake cause acne in teenagers?"
```

And hopefully get a bunch of relevant answers. You can also try other
variations of this, instead of "query" you can use:

* **fulltext** to use traditional fulltext search
* **hybrid** to combine fulltext and sematic using reciprocal rank fusion.

There are also versions of these commands to run over tab-separate CSV files,
useful for benchmarking. Please refer to the source code, or see the
nfcorpus-score.sh and scifact-score.sh scripts for examples of doing this
with datasets from BEIR.

## Using as Node module ##

We include a Makefile for building with Napi-rs and copying things around.

```
make buildemb
```

Builds target/release/warp.node with all the weights embedded into the binary,
not not have to deal with loading any files in production.  For day-to-day
development, you may prefer to instead use:


```
make build
```

Which loads the compressed weights from the "assets" dir, and deal with
symlinking "assets" into your Dash build tree by hand, as unfortunately Rust is
quite slow at embedding the binaries, and does it from scratch every time you
build.
