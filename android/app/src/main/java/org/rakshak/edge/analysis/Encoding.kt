package org.rakshak.edge.analysis

/**
 * Byte encoding of the dual-view student input (src/rakshak/models/encoding.py encode_bytes):
 * the UTF-8 bytes truncated to MAX_LEN and padded with PAD_ID. Each byte is one token, so the
 * int64 values are the byte values 0..255, and 256 is padding.
 */
const val MAX_LEN = 256
const val PAD_ID = 256

fun encodeBytes(text: String, maxLen: Int = MAX_LEN): LongArray {
    val raw = text.toByteArray(Charsets.UTF_8)
    val out = LongArray(maxLen) { PAD_ID.toLong() }
    val n = minOf(raw.size, maxLen)
    for (i in 0 until n) out[i] = (raw[i].toInt() and 0xFF).toLong()
    return out
}
