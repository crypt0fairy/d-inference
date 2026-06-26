// THROWAWAY harness: validate the pipeline model shard against REAL Llama
// weights on ONE machine. No networking, no second Mac.
//
//   swift run -c release shard-smoke <model-dir>
//
// where <model-dir> is a downloaded MLX Llama checkpoint, e.g.
//   hf download mlx-community/Llama-3.2-1B-Instruct-4bit --local-dir ~/m/llama-1b
//   swift run -c release shard-smoke ~/m/llama-1b
//
// What it does:
//   1. Build the FULL LlamaModel and load all weights (monolithic reference).
//   2. Build a 2-way layer split (head = [0, mid), tail = [mid, N)) via
//      LlamaPipelineShardLoader and load each shard's weights.
//   3. Run the same token sequence through both:
//        monolithic:  logits_full = model(tokens)
//        sharded:     h = head.embed(tokens); h = head.runOwnedLayers(h);
//                     h = tail.runOwnedLayers(h); logits_shard = tail.projectToLogits(h)
//   4. Assert argmax(logits_full) == argmax(logits_shard) at the last position,
//      and report max abs logit difference.
//
// This proves the loader (key filtering + index remap + 4-bit quantize) and the
// partial forward are correct against real weights. NOT a product.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

func fail(_ m: String) -> Never {
    FileHandle.standardError.write(Data("FAIL: \(m)\n".utf8)); exit(1)
}

let args = CommandLine.arguments
guard args.count >= 2 else { fail("usage: shard-smoke <model-dir>") }
let modelDir = URL(fileURLWithPath: args[1], isDirectory: true)

print("== pipeline shard smoke test ==")
print("model: \(modelDir.path)")

// This test validates the SHARDED forward against a MONOLITHIC reference loaded
// via LlamaPipelineShardLoader.loadFullModel — i.e. the Llama-family path only.
// GPT-OSS / other architectures whose full-model loader isn't wired here will
// fail at step 1 (e.g. unsupported `rope_scaling`); their sharded correctness is
// covered instead by `batch-shard-smoke` (sharded batched vs B=1 reference of
// the same sharded model, no monolithic load required).

// A short deterministic token sequence (no tokenizer needed for the math check).
let tokens = [1, 15043, 1234, 5678, 91, 42, 7]   // arbitrary valid ids
let ids = MLXArray(tokens.map { Int32($0) }, [1, tokens.count])
let noCache: [KVCache]? = nil

// ---- 1. Monolithic reference ----
print("\n[1/3] loading full model …")
let (full, N) = try LlamaPipelineShardLoader.loadFullModel(modelDir)
let mid = N / 2
print("layers: \(N)  split: head=[0,\(mid))  tail=[\(mid),\(N))")
let logitsFull = full(ids, cache: noCache)
logitsFull.eval()
print("    full logits shape: \(logitsFull.shape)")

// ---- 2. Sharded ----
print("[2/3] loading 2 shards …")
let (headShard, total1) = try LlamaPipelineShardLoader.loadFromDirectory(modelDir, start: 0, end: mid)
let (tailShard, total2) = try LlamaPipelineShardLoader.loadFromDirectory(modelDir, start: mid, end: N)
precondition(total1 == N && total2 == N)

// ---- 3. Sharded forward (no cache — single prefill pass, matches monolithic) ----
print("[3/3] running sharded forward …")
var h = headShard.embed(ids)
h = headShard.runOwnedLayers(h, cache: noCache)
h = tailShard.runOwnedLayers(h, cache: noCache)
let logitsShard = tailShard.projectToLogits(h)
logitsShard.eval()
print("    shard logits shape: \(logitsShard.shape)")

// ---- Compare ----
// The monolithic `full(ids)` returns logits for ALL positions ([1, seq, vocab]),
// while the sharded tail's `projectToLogits` is lastPositionOnly ([1, 1, vocab]).
// Both forwards only DECIDE the next token from the LAST position, so the
// apples-to-apples comparison is the full model's last-position slice vs the
// shard's single position. (Subtracting the raw tensors would broadcast
// [1,1,vocab] across all `seq` positions and report a meaningless max diff.)
func lastSlice(_ logits: MLXArray) -> MLXArray {
    // [1, seq, vocab] -> [1, 1, vocab] at the last position; [1,1,vocab] passes through.
    guard logits.ndim == 3, logits.dim(1) > 1 else { return logits }
    return logits[0..., (logits.dim(1) - 1)..., 0...]
}
func argmaxLast(_ logits: MLXArray) -> Int {
    let ids = argMax(logits, axis: logits.ndim - 1)
    ids.eval()
    return Int(ids.asArray(Int32.self).last ?? -1)
}
let fullLast = lastSlice(logitsFull)        // [1, 1, vocab]
let shardLast = lastSlice(logitsShard)      // [1, 1, vocab]
precondition(fullLast.shape == shardLast.shape,
             "aligned shapes expected, got full=\(fullLast.shape) shard=\(shardLast.shape)")

let aFull = argmaxLast(fullLast)
let aShard = argmaxLast(shardLast)

// Numerical closeness on the ALIGNED last-position logits. bf16 activations
// crossing the shard boundary accumulate small error over the layer stack, so
// compare both the absolute max diff and a scale-relative tolerance.
let diffArr = (fullLast - shardLast).abs()
diffArr.eval()
let maxDiff = diffArr.max().item(Float.self)
let scale = logitsFull.abs().max().item(Float.self)   // logit magnitude scale
let relDiff = scale > 0 ? maxDiff / scale : maxDiff

print("\nfull  argmax(last) = \(aFull)")
print("shard argmax(last) = \(aShard)")
print(String(format: "aligned max abs logit diff = %.4f  (scale %.1f, relative %.4f)", maxDiff, scale, relDiff))

// Pass criteria: identical next-token decision (the hard requirement) AND the
// aligned logits agree within a small relative tolerance (bf16 hop noise).
if aFull == aShard && relDiff < 0.02 {
    print("\nPASS: sharded forward matches monolithic (same next-token; aligned logits within \(String(format: "%.1f%%", relDiff*100)) relative).")
} else {
    fail("sharded output diverged from monolithic — argmax \(aFull) vs \(aShard), maxDiff \(maxDiff), relDiff \(relDiff)")
}
