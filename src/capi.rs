use crate::{
    match_centroids_raw, CachedEmbeddings, Embedder, EmbeddingCache, EmbeddingsCache,
    FileBackedIndex, IndexOptions,
};
#[cfg(feature = "capi-embed-cache")]
use crate::FileEmbeddingCache;
use anyhow::{anyhow, Result};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::ptr;
use std::slice;
use std::sync::Mutex;

const EMBEDDING_BLOB_MAGIC: [u8; 8] = *b"WPKD0001";

static GLOBAL_LAST_ERROR: Mutex<Option<CString>> = Mutex::new(None);

pub type WitchcraftEmbeddingCallback = Option<
    unsafe extern "C" fn(
        rowid: u64,
        user_data: *mut c_void,
        dst: *mut u8,
        dst_cap: usize,
        out_len: *mut usize,
    ) -> i32,
>;

struct CApiState {
    index: FileBackedIndex,
    device: candle_core::Device,
    embedder: Embedder,
    #[cfg(feature = "capi-embed-cache")]
    embedding_cache: FileEmbeddingCache,
    query_cache: EmbeddingsCache,
}

pub struct WitchcraftHandle {
    state: Mutex<CApiState>,
    last_error: Mutex<Option<CString>>,
}

#[repr(C)]
pub struct WitchcraftBytes {
    pub ptr: *mut u8,
    pub len: usize,
    pub status: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct WitchcraftSearchHit {
    pub rowid: u64,
    pub score: f32,
}

#[repr(C)]
pub struct WitchcraftSearchResults {
    pub ptr: *mut WitchcraftSearchHit,
    pub len: usize,
    pub status: i32,
}

fn error_cstring(message: impl ToString) -> CString {
    let message = message.to_string().replace('\0', " ");
    CString::new(message).unwrap_or_else(|_| CString::new("unknown error").unwrap())
}

fn set_global_error(message: impl ToString) {
    if let Ok(mut last_error) = GLOBAL_LAST_ERROR.lock() {
        *last_error = Some(error_cstring(message));
    }
}

fn clear_global_error() {
    if let Ok(mut last_error) = GLOBAL_LAST_ERROR.lock() {
        *last_error = None;
    }
}

fn set_handle_error(handle: &WitchcraftHandle, message: impl ToString) -> i32 {
    if let Ok(mut last_error) = handle.last_error.lock() {
        *last_error = Some(error_cstring(message));
    }
    -1
}

fn clear_handle_error(handle: &WitchcraftHandle) {
    if let Ok(mut last_error) = handle.last_error.lock() {
        *last_error = None;
    }
}

unsafe fn string_from_ptr(ptr: *const c_char, name: &str) -> Result<String> {
    if ptr.is_null() {
        return Err(anyhow!("{name} must not be null"));
    }
    Ok(CStr::from_ptr(ptr).to_str()?.to_string())
}

#[cfg(feature = "capi-embed-cache")]
unsafe fn embedding_cache_from_ptr(ptr: *const c_char) -> Result<FileEmbeddingCache> {
    if ptr.is_null() {
        return Ok(crate::default_embedding_cache());
    }

    let path = string_from_ptr(ptr, "embedding_cache_path")?;
    if path.is_empty() {
        return Err(anyhow!("embedding_cache_path must not be empty"));
    }
    Ok(FileEmbeddingCache::new(PathBuf::from(path)))
}

unsafe fn bytes_from_ptr<'a>(ptr: *const u8, len: usize, name: &str) -> Result<&'a [u8]> {
    if ptr.is_null() {
        if len == 0 {
            return Ok(&[]);
        }
        return Err(anyhow!("{name} must not be null"));
    }
    Ok(slice::from_raw_parts(ptr, len))
}

unsafe fn string_from_utf8_ptr(ptr: *const u8, len: usize, name: &str) -> Result<String> {
    Ok(std::str::from_utf8(bytes_from_ptr(ptr, len, name)?)?.to_string())
}

fn lens_for_body(body: &str) -> String {
    body.chars().count().to_string()
}

fn empty_bytes(status: i32) -> WitchcraftBytes {
    WitchcraftBytes {
        ptr: ptr::null_mut(),
        len: 0,
        status,
    }
}

fn bytes_result(bytes: Vec<u8>) -> WitchcraftBytes {
    if bytes.is_empty() {
        return empty_bytes(0);
    }
    let mut bytes = bytes.into_boxed_slice();
    let ptr = bytes.as_mut_ptr();
    let len = bytes.len();
    std::mem::forget(bytes);
    WitchcraftBytes {
        ptr,
        len,
        status: 0,
    }
}

fn empty_search_results(status: i32) -> WitchcraftSearchResults {
    WitchcraftSearchResults {
        ptr: ptr::null_mut(),
        len: 0,
        status,
    }
}

fn search_results_result(results: Vec<WitchcraftSearchHit>) -> WitchcraftSearchResults {
    if results.is_empty() {
        return empty_search_results(0);
    }
    let mut results = results.into_boxed_slice();
    let ptr = results.as_mut_ptr();
    let len = results.len();
    std::mem::forget(results);
    WitchcraftSearchResults {
        ptr,
        len,
        status: 0,
    }
}

fn write_u32(out: &mut Vec<u8>, value: usize) -> Result<()> {
    out.extend_from_slice(&u32::try_from(value)?.to_le_bytes());
    Ok(())
}

fn write_u64(out: &mut Vec<u8>, value: usize) -> Result<()> {
    out.extend_from_slice(&u64::try_from(value)?.to_le_bytes());
    Ok(())
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32> {
    let end = offset
        .checked_add(std::mem::size_of::<u32>())
        .ok_or_else(|| anyhow!("embedding blob offset overflow"))?;
    let chunk = bytes
        .get(*offset..end)
        .ok_or_else(|| anyhow!("truncated embedding blob"))?;
    *offset = end;
    Ok(u32::from_le_bytes(chunk.try_into()?))
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64> {
    let end = offset
        .checked_add(std::mem::size_of::<u64>())
        .ok_or_else(|| anyhow!("embedding blob offset overflow"))?;
    let chunk = bytes
        .get(*offset..end)
        .ok_or_else(|| anyhow!("truncated embedding blob"))?;
    *offset = end;
    Ok(u64::from_le_bytes(chunk.try_into()?))
}

fn read_bytes<'a>(bytes: &'a [u8], offset: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| anyhow!("embedding blob offset overflow"))?;
    let chunk = bytes
        .get(*offset..end)
        .ok_or_else(|| anyhow!("truncated embedding blob"))?;
    *offset = end;
    Ok(chunk)
}

fn encode_embedding_blob(embeddings: &CachedEmbeddings) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(
        EMBEDDING_BLOB_MAGIC.len()
            + 2 * std::mem::size_of::<u32>()
            + 2 * std::mem::size_of::<u64>()
            + embeddings.model.len()
            + embeddings.counts.len()
            + embeddings.embeddings.len(),
    );
    out.extend_from_slice(&EMBEDDING_BLOB_MAGIC);
    write_u32(&mut out, embeddings.model.len())?;
    write_u32(&mut out, embeddings.counts.len())?;
    write_u64(&mut out, embeddings.embedding_count)?;
    write_u64(&mut out, embeddings.embeddings.len())?;
    out.extend_from_slice(embeddings.model.as_bytes());
    out.extend_from_slice(embeddings.counts.as_bytes());
    out.extend_from_slice(&embeddings.embeddings);
    Ok(out)
}

fn decode_embedding_blob(bytes: &[u8]) -> Result<CachedEmbeddings> {
    let mut offset = 0usize;
    let magic = read_bytes(bytes, &mut offset, EMBEDDING_BLOB_MAGIC.len())?;
    if magic != EMBEDDING_BLOB_MAGIC {
        return Err(anyhow!("bad packed embedding blob magic"));
    }

    let model_len = read_u32(bytes, &mut offset)? as usize;
    let counts_len = read_u32(bytes, &mut offset)? as usize;
    let embedding_count = read_u64(bytes, &mut offset)? as usize;
    let embeddings_len = read_u64(bytes, &mut offset)? as usize;

    let model = String::from_utf8(read_bytes(bytes, &mut offset, model_len)?.to_vec())?;
    let counts = String::from_utf8(read_bytes(bytes, &mut offset, counts_len)?.to_vec())?;
    let embeddings = read_bytes(bytes, &mut offset, embeddings_len)?.to_vec();
    if offset != bytes.len() {
        return Err(anyhow!("packed embedding blob has trailing bytes"));
    }

    Ok(CachedEmbeddings {
        model,
        counts,
        embedding_count,
        embeddings,
    })
}

fn embed_for_capi(state: &CApiState, body_text: &str, lens: &str) -> Result<CachedEmbeddings> {
    #[cfg(feature = "capi-embed-cache")]
    {
        let hash = crate::document_cache_hash(body_text, lens);
        let (embeddings, _computed) = crate::load_or_compute_cached_embeddings(
            &state.embedding_cache,
            0,
            &hash,
            body_text,
            lens,
            &state.embedder,
        )?;
        Ok(embeddings)
    }
    #[cfg(not(feature = "capi-embed-cache"))]
    {
        crate::compute_cached_embeddings(&state.embedder, body_text, lens)
    }
}

fn embedding_blob_row_count(bytes: &[u8]) -> Result<u32> {
    let mut offset = 0usize;
    let magic = read_bytes(bytes, &mut offset, EMBEDDING_BLOB_MAGIC.len())?;
    if magic != EMBEDDING_BLOB_MAGIC {
        return Err(anyhow!("bad packed embedding blob magic"));
    }

    let model_len = read_u32(bytes, &mut offset)? as usize;
    let counts_len = read_u32(bytes, &mut offset)? as usize;
    let embedding_count = read_u64(bytes, &mut offset)?;
    let embeddings_len = read_u64(bytes, &mut offset)? as usize;

    read_bytes(bytes, &mut offset, model_len)?;
    read_bytes(bytes, &mut offset, counts_len)?;
    read_bytes(bytes, &mut offset, embeddings_len)?;
    if offset != bytes.len() {
        return Err(anyhow!("packed embedding blob has trailing bytes"));
    }

    Ok(embedding_count.try_into()?)
}

unsafe fn fetch_embedding_blob(
    callback: WitchcraftEmbeddingCallback,
    user_data: *mut c_void,
    rowid: u64,
) -> Result<CachedEmbeddings> {
    let Some(callback) = callback else {
        return Err(anyhow!("embedding callback must not be null"));
    };

    let mut len = 0usize;
    let status = callback(rowid, user_data, ptr::null_mut(), 0, &mut len);
    if status != 0 {
        return Err(anyhow!(
            "embedding callback failed for rowid {rowid}: {status}"
        ));
    }
    if len == 0 {
        return Err(anyhow!(
            "embedding callback returned empty data for rowid {rowid}"
        ));
    }

    let mut bytes = vec![0; len];
    let mut written = 0usize;
    let status = callback(
        rowid,
        user_data,
        bytes.as_mut_ptr(),
        bytes.len(),
        &mut written,
    );
    if status != 0 {
        return Err(anyhow!(
            "embedding callback failed for rowid {rowid}: {status}"
        ));
    }
    if written != bytes.len() {
        return Err(anyhow!(
            "embedding callback length changed for rowid {rowid}: expected {}, got {written}",
            bytes.len()
        ));
    }

    decode_embedding_blob(&bytes)
}

struct CallbackEmbeddingSource {
    callback: WitchcraftEmbeddingCallback,
    user_data: *mut c_void,
}

impl CallbackEmbeddingSource {
    fn new(callback: WitchcraftEmbeddingCallback, user_data: *mut c_void) -> Self {
        Self {
            callback,
            user_data,
        }
    }
}

impl EmbeddingCache for CallbackEmbeddingSource {
    fn get(&self, _hash: &str) -> Result<Option<CachedEmbeddings>> {
        Ok(None)
    }

    fn get_for_document(&self, rowid: u64, _hash: &str) -> Result<Option<CachedEmbeddings>> {
        unsafe { fetch_embedding_blob(self.callback, self.user_data, rowid).map(Some) }
    }

    fn put(&self, _hash: &str, _embeddings: &CachedEmbeddings) -> Result<()> {
        Err(anyhow!("callback embedding source is read-only"))
    }
}

unsafe fn handle_ref<'a>(handle: *mut WitchcraftHandle) -> Result<&'a WitchcraftHandle> {
    if handle.is_null() {
        return Err(anyhow!("handle must not be null"));
    }
    Ok(&*handle)
}

#[no_mangle]
pub unsafe extern "C" fn witchcraft_open(
    db_path: *const c_char,
    assets_path: *const c_char,
    embedding_cache_path: *const c_char,
) -> *mut WitchcraftHandle {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<*mut WitchcraftHandle> {
        let db_path = string_from_ptr(db_path, "db_path")?;
        let assets_path = string_from_ptr(assets_path, "assets_path")?;
        let device = crate::make_device();
        let embedder = Embedder::new(&device, &PathBuf::from(assets_path))?;
        #[cfg(feature = "capi-embed-cache")]
        let embedding_cache = embedding_cache_from_ptr(embedding_cache_path)?;
        #[cfg(not(feature = "capi-embed-cache"))]
        let _ = embedding_cache_path;
        let index = FileBackedIndex::new(PathBuf::from(db_path));
        let handle = WitchcraftHandle {
            state: Mutex::new(CApiState {
                index,
                device,
                embedder,
                #[cfg(feature = "capi-embed-cache")]
                embedding_cache,
                query_cache: EmbeddingsCache::new(128),
            }),
            last_error: Mutex::new(None),
        };
        Ok(Box::into_raw(Box::new(handle)))
    }));

    match result {
        Ok(Ok(handle)) => {
            clear_global_error();
            handle
        }
        Ok(Err(err)) => {
            set_global_error(err);
            ptr::null_mut()
        }
        Err(_) => {
            set_global_error("panic while opening witchcraft handle");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn witchcraft_close(handle: *mut WitchcraftHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub unsafe extern "C" fn witchcraft_embed(
    handle: *mut WitchcraftHandle,
    body_text: *const u8,
    body_text_len: usize,
) -> WitchcraftBytes {
    let handle = match handle_ref(handle) {
        Ok(handle) => handle,
        Err(err) => {
            set_global_error(err);
            return empty_bytes(-1);
        }
    };

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Vec<u8>> {
        let body_text = string_from_utf8_ptr(body_text, body_text_len, "body_text")?;
        let lens = lens_for_body(&body_text);
        let state = handle
            .state
            .lock()
            .map_err(|_| anyhow!("witchcraft handle lock poisoned"))?;
        let embeddings = embed_for_capi(&state, &body_text, &lens)?;
        encode_embedding_blob(&embeddings)
    }));

    match result {
        Ok(Ok(bytes)) => {
            clear_handle_error(handle);
            bytes_result(bytes)
        }
        Ok(Err(err)) => {
            set_handle_error(handle, err);
            empty_bytes(-1)
        }
        Err(_) => {
            set_handle_error(handle, "panic while embedding document");
            empty_bytes(-1)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn witchcraft_add(
    handle: *mut WitchcraftHandle,
    rowid: u64,
    embedding_blob: *const u8,
    embedding_blob_len: usize,
) -> i32 {
    let handle = match handle_ref(handle) {
        Ok(handle) => handle,
        Err(err) => {
            set_global_error(err);
            return -1;
        }
    };

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let embedding_blob = bytes_from_ptr(embedding_blob, embedding_blob_len, "embedding_blob")?;
        let rows = embedding_blob_row_count(embedding_blob)?;
        let state = handle
            .state
            .lock()
            .map_err(|_| anyhow!("witchcraft handle lock poisoned"))?;
        state.index.append_rowid_record(rowid, rows)?;
        Ok(())
    }));

    match result {
        Ok(Ok(())) => {
            clear_handle_error(handle);
            0
        }
        Ok(Err(err)) => set_handle_error(handle, err),
        Err(_) => set_handle_error(handle, "panic while adding document"),
    }
}

#[no_mangle]
pub unsafe extern "C" fn witchcraft_index(
    handle: *mut WitchcraftHandle,
    embedding_callback: WitchcraftEmbeddingCallback,
    user_data: *mut c_void,
) -> i32 {
    let handle = match handle_ref(handle) {
        Ok(handle) => handle,
        Err(err) => {
            set_global_error(err);
            return -1;
        }
    };

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let state = handle
            .state
            .lock()
            .map_err(|_| anyhow!("witchcraft handle lock poisoned"))?;
        let embeddings = CallbackEmbeddingSource::new(embedding_callback, user_data);
        crate::index_buffered_embeddings_with_options(
            &state.index,
            &embeddings,
            IndexOptions::default().force_flush(),
        )
    }));

    match result {
        Ok(Ok(())) => {
            clear_handle_error(handle);
            0
        }
        Ok(Err(err)) => set_handle_error(handle, err),
        Err(_) => set_handle_error(handle, "panic while indexing documents"),
    }
}

#[no_mangle]
pub unsafe extern "C" fn witchcraft_search(
    handle: *mut WitchcraftHandle,
    query: *const u8,
    query_len: usize,
    threshold: f32,
    top_k: usize,
) -> WitchcraftSearchResults {
    let handle = match handle_ref(handle) {
        Ok(handle) => handle,
        Err(err) => {
            set_global_error(err);
            return empty_search_results(-1);
        }
    };

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Vec<WitchcraftSearchHit>> {
        let query = string_from_utf8_ptr(query, query_len, "query")?;
        let mut state = handle
            .state
            .lock()
            .map_err(|_| anyhow!("witchcraft handle lock poisoned"))?;
        let CApiState {
            index,
            device,
            embedder,
            query_cache,
            ..
        } = &mut *state;
        let q = query.split_whitespace().collect::<Vec<_>>().join(" ");
        if q.len() <= 3 {
            return Ok(vec![]);
        }
        let qe = match query_cache.get(&q) {
            Some(existing) => existing,
            None => {
                let (qe, _) = embedder.embed(&q)?;
                let qe = qe.get(0)?;
                query_cache.put(&q, &qe);
                qe
            }
        };
        let generation_files = index.generation_files()?;
        let active = index.active_rowids()?;
        let scored = match_centroids_raw(
            &generation_files,
            &qe.to_device(device)?,
            &[],
            threshold,
            top_k,
        )?;
        let mut results = Vec::with_capacity(scored.len());
        let mut seen = std::collections::HashMap::<u64, bool>::new();
        for (score, rowid, _sub_idx) in scored {
            let rowid = rowid as u64;
            if active.get(&rowid).copied().unwrap_or(false) && seen.insert(rowid, true).is_none() {
                results.push(WitchcraftSearchHit { rowid, score });
                if results.len() == top_k {
                    break;
                }
            }
        }
        Ok(results)
    }));

    match result {
        Ok(Ok(results)) => {
            clear_handle_error(handle);
            search_results_result(results)
        }
        Ok(Err(err)) => {
            set_handle_error(handle, err);
            empty_search_results(-1)
        }
        Err(_) => {
            set_handle_error(handle, "panic while searching");
            empty_search_results(-1)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn witchcraft_bytes_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let slice = std::ptr::slice_from_raw_parts_mut(ptr, len);
    drop(Box::from_raw(slice));
}

#[no_mangle]
pub unsafe extern "C" fn witchcraft_search_results_free(ptr: *mut WitchcraftSearchHit, len: usize) {
    if ptr.is_null() {
        return;
    }
    let slice = std::ptr::slice_from_raw_parts_mut(ptr, len);
    drop(Box::from_raw(slice));
}

#[no_mangle]
pub unsafe extern "C" fn witchcraft_last_error(handle: *mut WitchcraftHandle) -> *const c_char {
    if handle.is_null() {
        return match GLOBAL_LAST_ERROR.lock() {
            Ok(last_error) => last_error
                .as_ref()
                .map(|message| message.as_ptr())
                .unwrap_or(ptr::null()),
            Err(_) => ptr::null(),
        };
    }

    match (*handle).last_error.lock() {
        Ok(last_error) => last_error
            .as_ref()
            .map(|message| message.as_ptr())
            .unwrap_or(ptr::null()),
        Err(_) => ptr::null(),
    }
}
