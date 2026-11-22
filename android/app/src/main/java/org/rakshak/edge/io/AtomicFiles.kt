package org.rakshak.edge.io

import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

/**
 * Moves [source] over [target]. It asks for an atomic rename first. Some storage (for example a
 * FUSE-backed external files directory) can refuse an atomic move; the move then falls back to a
 * plain replace, so a write that would otherwise fail still lands. Callers write [source] into the
 * same directory as [target], so either way the target is replaced by a complete file.
 */
internal fun moveReplacing(source: Path, target: Path) {
    try {
        Files.move(source, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
    } catch (_: AtomicMoveNotSupportedException) {
        Files.move(source, target, StandardCopyOption.REPLACE_EXISTING)
    }
}
