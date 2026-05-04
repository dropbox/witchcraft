//! Minimal V8 structured-clone (ValueSerializer) decoder.
//!
//! Handles the subset of types commonly found in Slack's IndexedDB:
//! objects, arrays, strings, numbers, booleans, null, undefined, Date, Map,
//! ArrayBuffer, ArrayBufferView (typed arrays).

use serde_json::Value;

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    fn read_byte(&mut self) -> Option<u8> {
        let b = self.data.get(self.pos).copied()?;
        self.pos += 1;
        Some(b)
    }

    fn read_varint(&mut self) -> Option<u64> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            let b = self.read_byte()? as u64;
            result |= (b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Some(result);
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }
    }

    fn read_double(&mut self) -> Option<f64> {
        if self.remaining() < 8 {
            return None;
        }
        let bytes: [u8; 8] = self.data[self.pos..self.pos + 8].try_into().ok()?;
        self.pos += 8;
        Some(f64::from_le_bytes(bytes))
    }

    fn read_bytes(&mut self, len: usize) -> Option<&'a [u8]> {
        if self.remaining() < len {
            return None;
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Some(slice)
    }
}

// V8 serialization tag bytes (from v8/src/objects/value-serializer.cc)
const TAG_TRUE: u8 = b'T';
const TAG_FALSE: u8 = b'F';
const TAG_NULL: u8 = b'0';
const TAG_UNDEFINED: u8 = b'_';
const TAG_INT32: u8 = b'I';
const TAG_UINT32: u8 = b'U';
const TAG_DOUBLE: u8 = b'N';
const TAG_UTF8_STRING: u8 = b'S';
const TAG_ONE_BYTE_STRING: u8 = b'"';
const TAG_TWO_BYTE_STRING: u8 = b'c';
const TAG_BEGIN_OBJECT: u8 = b'o';
const TAG_END_OBJECT: u8 = b'{';
const TAG_BEGIN_DENSE_ARRAY: u8 = b'A';
const TAG_END_DENSE_ARRAY: u8 = b'$';
const TAG_BEGIN_SPARSE_ARRAY: u8 = b'a';
const TAG_END_SPARSE_ARRAY: u8 = b'@';
const TAG_DATE: u8 = b'D';
const TAG_OBJECT_REFERENCE: u8 = b'^';
const TAG_BEGIN_MAP: u8 = b';';
const TAG_END_MAP: u8 = b':';
const TAG_BEGIN_SET: u8 = b'\'';
const TAG_END_SET: u8 = b',';
const TAG_REGEXP: u8 = b'R';
const TAG_ARRAY_BUFFER: u8 = b'B';
const TAG_ARRAY_BUFFER_VIEW: u8 = b'V';
const TAG_ARRAY_BUFFER_TRANSFER: u8 = b't';
const TAG_BIGINT: u8 = b'Z';
const TAG_THE_HOLE: u8 = b'-';
const TAG_PADDING: u8 = 0x00;
const TAG_VERIFY_OBJECT_COUNT: u8 = b'?';
// V8 header
const HEADER_TAG: u8 = 0xFF;

/// Deserialize a Chrome IndexedDB record value into a JSON value.
///
/// The stored format is:
///   <varint: IDB record version>
///   0xFF <blink_ssv_version>
///   [if blink_version >= 17: 0xFE <u64le: trailer_offset> <u32le: trailer_count>]
///   0xFF <v8_version>
///   <V8 structured-clone body>
///
/// For raw V8 payloads (starting directly with 0xFF), the envelope is skipped.
pub fn deserialize(data: &[u8]) -> Option<Value> {
    let mut r = Reader::new(data);

    // Try to detect Chrome IndexedDB envelope vs raw V8.
    // If the first byte is 0xFF, it's raw V8.  Otherwise, skip the IDB envelope.
    if r.peek()? != HEADER_TAG {
        // Skip the IDB record version varint
        let _idb_version = r.read_varint()?;
    }

    // Read Blink SSV header: 0xFF <blink_version>
    if r.read_byte()? != HEADER_TAG {
        return None;
    }
    let blink_version = r.read_varint()?;

    // Blink SSV version >= 17 may have a trailer offset block (0xFE + 12 bytes).
    // The trailer is optional — only consume it if the tag is present.
    if blink_version >= 17 && r.peek() == Some(0xFE) {
        r.read_byte(); // consume 0xFE
        // Skip uint64le trailer_offset + uint32le trailer_count = 12 bytes
        r.read_bytes(12)?;
    }

    // Read V8 structured-clone header: 0xFF <v8_version>
    if r.read_byte()? != HEADER_TAG {
        return None;
    }
    let v8_version = r.read_varint()?;

    let mut objects: Vec<Value> = Vec::new();
    read_value(&mut r, &mut objects, v8_version)
}

/// Deserialize a Chrome IndexedDB blob file into a JSON value.
///
/// Blob files use an IDB value wrapper:
///   0xFF <wrapper_version: varint> <tag>
///   - tag 0x01: no wrapping, raw SSV follows
///   - tag 0x02/0x03: Snappy-compressed SSV data follows (from byte 3)
///
/// The decompressed data is then parsed with the normal SSV+V8 envelope.
pub fn deserialize_blob(data: &[u8]) -> Option<Value> {
    if data.len() < 3 || data[0] != HEADER_TAG {
        return deserialize(data);
    }

    let mut r = Reader::new(data);
    r.read_byte(); // skip 0xFF
    let _wrapper_version = r.read_varint()?;
    let tag = r.read_byte()?;

    match tag {
        0x01 => {
            // kDone — raw SSV+V8 data follows at current position
            deserialize(&data[r.pos..])
        }
        0x02 | 0x03 => {
            // Snappy-compressed SSV+V8 data from byte 3
            let decompressed = snap::raw::Decoder::new()
                .decompress_vec(&data[r.pos..])
                .ok()?;
            deserialize(&decompressed)
        }
        _ => {
            // Unknown tag — try normal deserialization as fallback
            deserialize(data)
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn read_value(r: &mut Reader, objects: &mut Vec<Value>, v8_version: u64) -> Option<Value> {
    let tag = r.read_byte()?;
    match tag {
        TAG_PADDING => read_value(r, objects, v8_version), // skip padding, read next
        TAG_VERIFY_OBJECT_COUNT => {
            let _count = r.read_varint()?;
            read_value(r, objects, v8_version) // skip verify count, read next
        }
        TAG_TRUE => Some(Value::Bool(true)),
        TAG_FALSE => Some(Value::Bool(false)),
        TAG_NULL | TAG_UNDEFINED => Some(Value::Null),
        TAG_THE_HOLE => Some(Value::Null), // sparse array hole
        TAG_INT32 => {
            let v = r.read_varint()? as u32;
            // Zigzag decode
            let n = ((v >> 1) as i32) ^ -((v & 1) as i32);
            Some(Value::Number(serde_json::Number::from(n)))
        }
        TAG_UINT32 => {
            let v = r.read_varint()? as u64;
            Some(Value::Number(serde_json::Number::from(v)))
        }
        TAG_DOUBLE => {
            let v = r.read_double()?;
            serde_json::Number::from_f64(v)
                .map(Value::Number)
                .or(Some(Value::Null))
        }
        TAG_DATE => {
            let millis = r.read_double()?;
            Some(Value::Number(
                serde_json::Number::from_f64(millis).unwrap_or(serde_json::Number::from(0)),
            ))
        }
        TAG_UTF8_STRING => {
            let len = r.read_varint()? as usize;
            let bytes = r.read_bytes(len)?;
            let s = String::from_utf8_lossy(bytes);
            Some(Value::String(s.into_owned()))
        }
        TAG_ONE_BYTE_STRING => {
            let len = r.read_varint()? as usize;
            let bytes = r.read_bytes(len)?;
            // Latin-1 encoding — bytes 0-255 map to Unicode code points directly.
            let s: String = bytes.iter().map(|&b| b as char).collect();
            Some(Value::String(s))
        }
        TAG_TWO_BYTE_STRING => {
            let byte_length = r.read_varint()? as usize;
            let bytes = r.read_bytes(byte_length)?;
            // V8 TwoByteStrings are always UTF-16LE. The varint stores the
            // byte count which must be even.
            if byte_length % 2 != 0 {
                return None; // corrupt data
            }
            let chars: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let s = String::from_utf16_lossy(&chars);
            Some(Value::String(s))
        }
        TAG_BEGIN_OBJECT => {
            let idx = objects.len();
            objects.push(Value::Null); // placeholder for back-references
            let mut map = serde_json::Map::new();
            loop {
                match r.peek()? {
                    TAG_END_OBJECT => {
                        r.read_byte();
                        let _num_properties = r.read_varint()?;
                        break;
                    }
                    _ => {}
                }
                let key = read_value(r, objects, v8_version)?;
                let val = read_value(r, objects, v8_version)?;
                let key_str = match key {
                    Value::String(s) => s,
                    Value::Number(n) => n.to_string(),
                    _ => format!("{}", key),
                };
                map.insert(key_str, val);
            }
            let obj = Value::Object(map);
            objects[idx] = obj.clone();
            Some(obj)
        }
        TAG_BEGIN_DENSE_ARRAY => {
            let len = r.read_varint()? as usize;
            let idx = objects.len();
            objects.push(Value::Null); // placeholder
            let mut arr = Vec::with_capacity(len.min(10_000));
            for _ in 0..len {
                arr.push(read_value(r, objects, v8_version)?);
            }
            // After elements: named properties (rare) then end tag
            loop {
                match r.peek()? {
                    TAG_END_DENSE_ARRAY => {
                        r.read_byte();
                        let _num_properties = r.read_varint()?;
                        let _length = r.read_varint()?;
                        break;
                    }
                    _ => {
                        // Skip named property key+value
                        read_value(r, objects, v8_version)?;
                        read_value(r, objects, v8_version)?;
                    }
                }
            }
            let val = Value::Array(arr);
            objects[idx] = val.clone();
            Some(val)
        }
        TAG_BEGIN_SPARSE_ARRAY => {
            let len = r.read_varint()? as usize;
            let idx = objects.len();
            objects.push(Value::Null);
            let mut arr = vec![Value::Null; len.min(10_000)];
            loop {
                match r.peek()? {
                    TAG_END_SPARSE_ARRAY => {
                        r.read_byte();
                        let _num_properties = r.read_varint()?;
                        let _length = r.read_varint()?;
                        break;
                    }
                    _ => {
                        let key = read_value(r, objects, v8_version)?;
                        let val = read_value(r, objects, v8_version)?;
                        if let Some(i) = key.as_u64() {
                            if (i as usize) < arr.len() {
                                arr[i as usize] = val;
                            }
                        }
                    }
                }
            }
            let val = Value::Array(arr);
            objects[idx] = val.clone();
            Some(val)
        }
        TAG_BEGIN_MAP => {
            let idx = objects.len();
            objects.push(Value::Null);
            let mut map = serde_json::Map::new();
            loop {
                match r.peek()? {
                    TAG_END_MAP => {
                        r.read_byte();
                        let _num_entries = r.read_varint()?;
                        break;
                    }
                    _ => {
                        let key = read_value(r, objects, v8_version)?;
                        let val = read_value(r, objects, v8_version)?;
                        let key_str = match key {
                            Value::String(s) => s,
                            Value::Number(n) => n.to_string(),
                            _ => format!("{}", key),
                        };
                        map.insert(key_str, val);
                    }
                }
            }
            let val = Value::Object(map);
            objects[idx] = val.clone();
            Some(val)
        }
        TAG_BEGIN_SET => {
            let idx = objects.len();
            objects.push(Value::Null);
            let mut arr = Vec::new();
            loop {
                match r.peek()? {
                    TAG_END_SET => {
                        r.read_byte();
                        let _num_entries = r.read_varint()?;
                        break;
                    }
                    _ => {
                        arr.push(read_value(r, objects, v8_version)?);
                    }
                }
            }
            let val = Value::Array(arr);
            objects[idx] = val.clone();
            Some(val)
        }
        TAG_OBJECT_REFERENCE => {
            let id = r.read_varint()? as usize;
            objects.get(id).cloned()
        }
        TAG_REGEXP => {
            // pattern (string) + flags (varint)
            let pattern = read_value(r, objects, v8_version)?;
            let _flags = r.read_varint()?;
            Some(pattern)
        }
        TAG_ARRAY_BUFFER => {
            let byte_length = r.read_varint()? as usize;
            let data = r.read_bytes(byte_length)?;
            let val = Value::String(hex_encode(data));
            objects.push(val.clone());
            // V8 pairs ArrayBuffer + ArrayBufferView: if the next tag is 'V',
            // consume the view metadata and return the buffer data.
            if r.peek() == Some(TAG_ARRAY_BUFFER_VIEW) {
                skip_array_buffer_view(r, v8_version);
            }
            Some(val)
        }
        TAG_ARRAY_BUFFER_VIEW => {
            // Orphan view (no preceding buffer) — skip metadata
            skip_array_buffer_view(r, v8_version);
            Some(Value::Null)
        }
        TAG_ARRAY_BUFFER_TRANSFER => {
            let _transfer_id = r.read_varint()?;
            Some(Value::Null)
        }
        TAG_BIGINT => {
            let bitfield = r.read_varint()?;
            let digit_count = (bitfield >> 1) as usize;
            r.read_bytes(digit_count * 8)?;
            Some(Value::Null) // Skip BigInts
        }
        _ => {
            // Unknown tag — can't continue reliably.
            None
        }
    }
}

fn skip_array_buffer_view(r: &mut Reader, v8_version: u64) {
    // ArrayBufferView format: tag(varint) + byte_offset(varint) + byte_length(varint)
    // + flags(varint) if v8_version >= 14
    let _ = r.read_byte(); // consume 'V' tag
    let _ = r.read_varint(); // view type (Uint8Array='B', etc.)
    let _ = r.read_varint(); // byte_offset
    let _ = r.read_varint(); // byte_length
    if v8_version >= 14 {
        let _ = r.read_varint(); // flags
    }
}
