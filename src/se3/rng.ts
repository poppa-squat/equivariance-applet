/** Deterministic mulberry32 PRNG for fixed demo weights. */
export function mulberry32(seed: number): () => number {
  let t = seed >>> 0
  return () => {
    t = (t + 0x6d2b79f5) >>> 0
    let r = Math.imul(t ^ (t >>> 15), 1 | t)
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}

export function randn(rng: () => number): number {
  // Box–Muller
  const u = Math.max(rng(), 1e-12)
  const v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

export function randnArray(n: number, rng: () => number, scale = 1): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = randn(rng) * scale
  return out
}
