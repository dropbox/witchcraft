#ifndef WITCHCRAFT_H
#define WITCHCRAFT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WitchcraftHandle WitchcraftHandle;

typedef struct WitchcraftRowids {
    uint64_t *ptr;
    size_t len;
    int32_t status;
} WitchcraftRowids;

typedef struct WitchcraftBytes {
    uint8_t *ptr;
    size_t len;
    int32_t status;
} WitchcraftBytes;

typedef int32_t (*WitchcraftEmbeddingCallback)(
    uint64_t rowid,
    void *user_data,
    uint8_t *dst,
    size_t dst_cap,
    size_t *out_len
);

WitchcraftHandle *witchcraft_open(const char *db_path, const char *assets_path);
void witchcraft_close(WitchcraftHandle *handle);

WitchcraftBytes witchcraft_embed(
    WitchcraftHandle *handle,
    const uint8_t *body_text,
    size_t body_text_len
);

int32_t witchcraft_add(
    WitchcraftHandle *handle,
    uint64_t rowid,
    uint32_t rows
);

int32_t witchcraft_index(
    WitchcraftHandle *handle,
    WitchcraftEmbeddingCallback embedding_callback,
    void *user_data
);

WitchcraftRowids witchcraft_search(
    WitchcraftHandle *handle,
    const uint8_t *query,
    size_t query_len,
    float threshold,
    size_t top_k
);

void witchcraft_bytes_free(uint8_t *ptr, size_t len);
void witchcraft_rowids_free(uint64_t *ptr, size_t len);
const char *witchcraft_last_error(WitchcraftHandle *handle);

#ifdef __cplusplus
}
#endif

#endif
