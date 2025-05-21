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
  XTRLinear layer, and exports into xtr.safetensors

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
mkdir documents
python createdocs.py datasets/nfcorpus.tsv
```

With all the documents in place, it is time to create the index:

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

And hopefully get a bunch of relevant answers.

## Caveats ##

Currently the vector compression that XTR-Warp uses is not implemented,
and we also keep the document vectors around outside the index, so the
mydb.sqlite file will get quite large. This is work in progress, but 
in general the index is going to need more space that than the raw
indexed text, due to storing a full embedding per token of input.

XTR-Warp needs three hyperparameters to be tuned for the dataset size,

* k in k-means
* k' which is the number of nearby centroids to oversample on lookup
* t' which determines the cumulative cluster size from where scores
  are imputed

These are currently all hardcoded to work well only with nfcorpus.
