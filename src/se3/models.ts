import {
  type Features,
  type Vec3,
  emptyFeatures,
  vCross,
  vDot,
  vNorm,
  vScale,
  vSub,
} from './math'
import { mulberry32, randnArray } from './rng'

const SEED = 0x53e30001

function tanh(x: number): number {
  if (x > 20) return 1
  if (x < -20) return -1
  const e = Math.exp(2 * x)
  return (e - 1) / (e + 1)
}

function relu(x: number): number {
  return x > 0 ? x : 0
}

function softplus(x: number): number {
  if (x > 20) return x
  return Math.log1p(Math.exp(x))
}

// ─── MLP baseline (not SE(3)-aware) ─────────────────────────────────────────

interface MlpWeights {
  /** Input: 3N coords → hidden */
  W1: Float64Array
  b1: Float64Array
  /** Hidden → hidden */
  W2: Float64Array
  b2: Float64Array
  /** Hidden → N scalars + 3N vector components */
  W3: Float64Array
  b3: Float64Array
  n: number
  h: number
}

function initMlp(n: number, h: number, seed: number): MlpWeights {
  const rng = mulberry32(seed)
  const inDim = 3 * n
  const outDim = n + 3 * n
  const scale1 = Math.sqrt(2 / inDim)
  const scale2 = Math.sqrt(2 / h)
  const scale3 = Math.sqrt(1 / h)
  return {
    W1: randnArray(h * inDim, rng, scale1),
    b1: randnArray(h, rng, 0.05),
    W2: randnArray(h * h, rng, scale2),
    b2: randnArray(h, rng, 0.05),
    W3: randnArray(outDim * h, rng, scale3),
    b3: randnArray(outDim, rng, 0.02),
    n,
    h,
  }
}

function mlpForward(w: MlpWeights, points: Vec3[]): Features {
  const { n, h } = w
  const x = new Float64Array(3 * n)
  for (let i = 0; i < n; i++) {
    x[i * 3] = points[i][0]
    x[i * 3 + 1] = points[i][1]
    x[i * 3 + 2] = points[i][2]
  }

  const h1 = new Float64Array(h)
  for (let i = 0; i < h; i++) {
    let s = w.b1[i]
    const row = i * 3 * n
    for (let j = 0; j < 3 * n; j++) s += w.W1[row + j] * x[j]
    h1[i] = relu(s)
  }

  const h2 = new Float64Array(h)
  for (let i = 0; i < h; i++) {
    let s = w.b2[i]
    const row = i * h
    for (let j = 0; j < h; j++) s += w.W2[row + j] * h1[j]
    h2[i] = tanh(s)
  }

  const outDim = n + 3 * n
  const y = new Float64Array(outDim)
  for (let i = 0; i < outDim; i++) {
    let s = w.b3[i]
    const row = i * h
    for (let j = 0; j < h; j++) s += w.W3[row + j] * h2[j]
    y[i] = s
  }

  const f = emptyFeatures(n)
  for (let i = 0; i < n; i++) f.scalars[i] = y[i]
  for (let i = 0; i < 3 * n; i++) f.vectors[i] = y[n + i] * 0.55
  return f
}

// ─── SE(3)-equivariant message passing toy ──────────────────────────────────

interface EqWeights {
  /** Edge MLP on [‖rij‖, softplus(‖rij‖)] → scalar messages (2 → h → 2 channels) */
  eW1: Float64Array
  eb1: Float64Array
  eW2: Float64Array
  eb2: Float64Array
  /** Node scalar update: aggregated scalars → scalar out */
  sW: Float64Array
  sb: Float64Array
  /** Vector gate scalars → coefficients for relative vectors + cross terms */
  vGate: Float64Array
  vb: Float64Array
  n: number
  h: number
}

function initEq(n: number, h: number, seed: number): EqWeights {
  const rng = mulberry32(seed ^ 0x0e03)
  return {
    eW1: randnArray(h * 2, rng, 0.8),
    eb1: randnArray(h, rng, 0.1),
    eW2: randnArray(2 * h, rng, 0.5),
    eb2: randnArray(2, rng, 0.05),
    sW: randnArray(n * 3, rng, 0.4), // [sum_phi, sum_psi, self_bias_slot] loosely
    sb: randnArray(n, rng, 0.05),
    vGate: randnArray(4, rng, 0.6),
    vb: randnArray(2, rng, 0.1),
    n,
    h,
  }
}

/**
 * Message passing on relative vectors r_ij = x_j − x_i:
 * - scalars from invariant distances
 * - vectors = linear combos of r_ij (+ optional cross products for chirality)
 * Output transforms correctly under SE(3) by construction.
 */
function eqForward(w: EqWeights, points: Vec3[]): Features {
  const n = points.length
  const f = emptyFeatures(n)
  const { h } = w

  // Per-node accumulators
  const scalarAcc = new Float64Array(n)
  const scalarAcc2 = new Float64Array(n)
  const vecAcc = new Float64Array(n * 3)
  const crossAcc = new Float64Array(n * 3)

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue
      const rij = vSub(points[j], points[i])
      const d = vNorm(rij)
      const feats = [d, softplus(d - 1)]

      // Edge MLP → two invariant message channels
      const hid = new Float64Array(h)
      for (let k = 0; k < h; k++) {
        let s = w.eb1[k]
        s += w.eW1[k * 2] * feats[0] + w.eW1[k * 2 + 1] * feats[1]
        hid[k] = relu(s)
      }
      let m0 = w.eb2[0]
      let m1 = w.eb2[1]
      for (let k = 0; k < h; k++) {
        m0 += w.eW2[k] * hid[k]
        m1 += w.eW2[h + k] * hid[k]
      }
      m0 = tanh(m0)
      m1 = tanh(m1)

      scalarAcc[i] += m0
      scalarAcc2[i] += m1

      // Equivariant vector messages: φ(d) * r_ij
      const inv = 1 / Math.max(d, 1e-6)
      const phi = m0 * inv
      const psi = m1 * inv
      vecAcc[i * 3] += phi * rij[0]
      vecAcc[i * 3 + 1] += phi * rij[1]
      vecAcc[i * 3 + 2] += phi * rij[2]

      // Chirality-aware SO(3)-equivariant cross term using a fixed neighbor pair flavor:
      // accumulate ψ * (r_ij × e_z_local-ish) — use another relative direction via j-mean.
      // Simpler: ψ * r_ij already; add cross with a global-ish but input-derived axis
      // built only from relatives: cross of two messages. For each edge use cross(rij, sum_k rik)
      // deferred — collect later.
      crossAcc[i * 3] += psi * rij[0]
      crossAcc[i * 3 + 1] += psi * rij[1]
      crossAcc[i * 3 + 2] += psi * rij[2]
    }
  }

  // Second pass: cross products of the two accumulated equivariant channels
  // (cross of two vectors that each transform as vectors → transforms as vector)
  const g0 = tanh(w.vGate[0] + w.vb[0])
  const g1 = tanh(w.vGate[1] + w.vb[1])
  const g2 = tanh(w.vGate[2])
  const g3 = tanh(w.vGate[3])

  for (let i = 0; i < n; i++) {
    const v1: Vec3 = [vecAcc[i * 3], vecAcc[i * 3 + 1], vecAcc[i * 3 + 2]]
    const v2: Vec3 = [crossAcc[i * 3], crossAcc[i * 3 + 1], crossAcc[i * 3 + 2]]
    const c = vCross(v1, v2)

    f.vectors[i * 3] = g0 * v1[0] + g1 * v2[0] + g2 * c[0]
    f.vectors[i * 3 + 1] = g0 * v1[1] + g1 * v2[1] + g2 * c[1]
    f.vectors[i * 3 + 2] = g0 * v1[2] + g1 * v2[2] + g2 * c[2]

    // Scale for readable glyphs
    f.vectors[i * 3] *= 0.35
    f.vectors[i * 3 + 1] *= 0.35
    f.vectors[i * 3 + 2] *= 0.35

    const self = vDot(v1, v1)
    f.scalars[i] =
      tanh(
        w.sb[i] +
          w.sW[i * 3] * scalarAcc[i] +
          w.sW[i * 3 + 1] * scalarAcc2[i] +
          w.sW[i * 3 + 2] * softplus(self) * g3,
      ) * 1.2
  }

  return f
}

// ─── Public API ─────────────────────────────────────────────────────────────

let mlpW: MlpWeights | null = null
let eqW: EqWeights | null = null

function ensure(n: number) {
  if (!mlpW || mlpW.n !== n) mlpW = initMlp(n, 32, SEED)
  if (!eqW || eqW.n !== n) eqW = initEq(n, 16, SEED)
}

export function forwardMlp(points: Vec3[]): Features {
  ensure(points.length)
  return mlpForward(mlpW!, points)
}

export function forwardEquivariant(points: Vec3[]): Features {
  ensure(points.length)
  return eqForward(eqW!, points)
}

/** Tiny helper for unit tests / sanity checks. */
export function scaleFeatures(f: Features, s: number): Features {
  const out = emptyFeatures(f.scalars.length)
  for (let i = 0; i < f.scalars.length; i++) out.scalars[i] = f.scalars[i] * s
  for (let i = 0; i < f.vectors.length; i++) out.vectors[i] = f.vectors[i] * s
  return out
}

export { vScale }
