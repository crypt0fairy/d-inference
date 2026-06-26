/// LlamaShardAdapter -- bridges the mlx-swift-lm fork's `LlamaPipelineShard`
/// (which owns this rank's actual transformer layers + weights) to ProviderCore's
/// `PipelineModelShard` protocol that `DistributedInferenceEngine` /
/// `PipelineRunner` drive.
///
/// The fork provides the model-internal slice (embed / owned layers / project),
/// keeping a per-owned-layer KV cache across decode steps; this adapter just maps
/// the protocol calls onto it and tokens→MLXArray.
///
/// This is the concrete shard that makes a real model run split across the
/// cluster. The loader (`LlamaPipelineShardLoader.load`) builds one from a model
/// directory + the rank's `LayerInterval`.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

public final class LlamaShardAdapter: PipelineModelShard, @unchecked Sendable {
    private let shard: LlamaPipelineShard
    public let totalLayers: Int
    public let ownedInterval: LayerInterval

    /// One KV cache per owned layer, persisted across decode steps so attention
    /// sees prior tokens. Created lazily on first use.
    private var kvCaches: [KVCache]?

    /// Batched KV caches (one BatchKVCache per owned layer) for the
    /// continuous-batching path. Allocated by `beginBatch`. Separate from the
    /// B=1 `kvCaches` so the two decode paths never share state.
    private var batchedCaches: [KVCache]?

    public init(shard: LlamaPipelineShard, ownedInterval: LayerInterval) {
        self.shard = shard
        self.totalLayers = shard.range.totalLayers
        self.ownedInterval = ownedInterval
    }

    /// Build the adapter (and load this rank's weights) from a model directory.
    /// Parses config.json + quantization inside the fork loader, so ProviderCore
    /// needs no access to MLXLLM's internal config fields.
    public static func load(directory: URL, interval: LayerInterval) throws -> LlamaShardAdapter {
        let (shard, _) = try LlamaPipelineShardLoader.loadFromDirectory(
            directory, start: interval.start, end: interval.end)
        return LlamaShardAdapter(shard: shard, ownedInterval: interval)
    }

    // MARK: - PipelineModelShard

    public func embed(tokens: [Int]) -> MLXArray {
        // Shape [1, seqLen] of Int32 token ids.
        let ids = MLXArray(tokens.map { Int32($0) }, [1, tokens.count])
        return shard.embed(ids)
    }

    public func runOwnedLayers(_ hidden: MLXArray) -> MLXArray {
        // evalEvery: 4 keeps each Metal command buffer short so a tight-memory
        // node can't run a single buffer long enough to trip the ~5s GPU
        // watchdog during prefill.
        shard.runOwnedLayers(hidden, cache: caches(), evalEvery: 4)
    }

    public func projectToLogits(_ hidden: MLXArray) -> MLXArray {
        shard.projectToLogits(hidden)
    }

    // MARK: - Batched path

    public func embedBatch(rows: [[Int]], leftPadding: [Int]) -> MLXArray {
        // Left-pad each row to the batch's max length so all rows share one
        // [B, maxLen] matrix. Phase 1: rows are equal-length ⇒ no padding.
        let maxLen = rows.map(\.count).max() ?? 0
        var flat = [Int32]()
        flat.reserveCapacity(rows.count * maxLen)
        for (b, row) in rows.enumerated() {
            let pad = leftPadding.indices.contains(b) ? leftPadding[b] : (maxLen - row.count)
            flat.append(contentsOf: repeatElement(Int32(0), count: pad))
            flat.append(contentsOf: row.map { Int32($0) })
            // Right-fill any remainder (defensive; equal-length rows hit neither).
            let filled = pad + row.count
            if filled < maxLen { flat.append(contentsOf: repeatElement(Int32(0), count: maxLen - filled)) }
        }
        let ids = MLXArray(flat, [rows.count, maxLen])
        return shard.embed(ids)
    }

    public func runOwnedLayersBatched(_ hidden: MLXArray) -> MLXArray {
        shard.runOwnedLayers(hidden, cache: batchedCachesOrEmpty(), evalEvery: 4)
    }

    public func projectToLogitsBatched(_ hidden: MLXArray) -> MLXArray {
        shard.projectToLogits(hidden)
    }

    public func beginBatch(leftPadding: [Int]) {
        batchedCaches = shard.makeBatchedCaches(leftPadding: leftPadding)
    }

    public func filterBatch(keepIndices: [Int]) {
        guard let caches = batchedCaches else { return }
        let idx = MLXArray(keepIndices.map { Int32($0) }, [keepIndices.count])
        for c in caches {
            (c as? BatchedCache)?.filterBatched(batchIndices: idx)
        }
    }

    public func admitPrefill(_ hidden: MLXArray, leftPadding: [Int]) -> MLXArray {
        // Prefill the admitted rows through a SEPARATE temporary batched cache,
        // then append those rows onto the live caches (extend). Returns the
        // post-owned-layers state for the admitted rows so the tail can sample.
        let admitted = shard.makeBatchedCaches(leftPadding: leftPadding)
        let out = shard.runOwnedLayers(hidden, cache: admitted, evalEvery: 4)
        if let live = batchedCaches {
            for (i, c) in live.enumerated() {
                if let bc = c as? BatchedCache, i < admitted.count, let other = admitted[i] as? BatchedCache {
                    bc.extendBatched(other)
                }
            }
        }
        return out
    }

    // MARK: - KV cache

    private func caches() -> [KVCache] {
        if let kvCaches { return kvCaches }
        let made = (0 ..< shard.ownedLayerCount).map { _ in KVCacheSimple() as KVCache }
        kvCaches = made
        return made
    }

    private func batchedCachesOrEmpty() -> [KVCache] {
        batchedCaches ?? []
    }
}
