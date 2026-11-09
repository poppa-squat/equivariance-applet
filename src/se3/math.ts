/** SE(3) and SO(3) helpers — explicit teaching math, no heavy libs. */

export type Vec3 = [number, number, number]
export type Mat3 = [number, number, number, number, number, number, number, number, number]

export interface SE3 {
  /** Row-major 3×3 rotation */
  R: Mat3
  t: Vec3
}

export const identitySE3 = (): SE3 => ({
  R: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  t: [0, 0, 0],
})

export function cloneSE3(g: SE3): SE3 {
  return { R: [...g.R] as Mat3, t: [...g.t] as Vec3 }
}

export function vAdd(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

export function vSub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

export function vScale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s]
}

export function vDot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

export function vNorm(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2])
}

export function vCross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

export function applyR(R: Mat3, v: Vec3): Vec3 {
  return [
    R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
    R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
    R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
  ]
}

export function applySE3(g: SE3, x: Vec3): Vec3 {
  return vAdd(applyR(g.R, x), g.t)
}

export function transformCloud(g: SE3, points: Vec3[]): Vec3[] {
  return points.map((p) => applySE3(g, p))
}

/** Axis-angle → rotation matrix (Rodrigues). */
export function axisAngleToR(axis: Vec3, angle: number): Mat3 {
  const n = vNorm(axis)
  if (n < 1e-12 || Math.abs(angle) < 1e-12) {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1]
  }
  const ux = axis[0] / n
  const uy = axis[1] / n
  const uz = axis[2] / n
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  const t = 1 - c
  return [
    t * ux * ux + c,
    t * ux * uy - s * uz,
    t * ux * uz + s * uy,
    t * ux * uy + s * uz,
    t * uy * uy + c,
    t * uy * uz - s * ux,
    t * ux * uz - s * uy,
    t * uy * uz + s * ux,
    t * uz * uz + c,
  ]
}

export function matMul3(A: Mat3, B: Mat3): Mat3 {
  const out = new Array(9) as number[]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i * 3 + j] =
        A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j]
    }
  }
  return out as Mat3
}

export function composeSE3(a: SE3, b: SE3): SE3 {
  // a ∘ b : x ↦ a.R (b.R x + b.t) + a.t
  return {
    R: matMul3(a.R, b.R),
    t: vAdd(applyR(a.R, b.t), a.t),
  }
}

export function randomSE3(rng: () => number): SE3 {
  const axis: Vec3 = [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1]
  const angle = (rng() * 2 - 1) * Math.PI
  const t: Vec3 = [(rng() - 0.5) * 1.4, (rng() - 0.5) * 1.4, (rng() - 0.5) * 1.0]
  return { R: axisAngleToR(axis, angle), t }
}

/** Extract Euler-ish yaw/pitch/roll display from R (ZYX). */
export function RtoEuler(R: Mat3): Vec3 {
  const sy = Math.hypot(R[0], R[3])
  let x: number, y: number, z: number
  if (sy > 1e-6) {
    x = Math.atan2(R[7], R[8])
    y = Math.atan2(-R[6], sy)
    z = Math.atan2(R[3], R[0])
  } else {
    x = Math.atan2(-R[5], R[4])
    y = Math.atan2(-R[6], sy)
    z = 0
  }
  return [x, y, z]
}

export interface Features {
  /** Per-point invariant scalars */
  scalars: Float64Array
  /** Per-point equivariant vectors, length 3N */
  vectors: Float64Array
}

export function emptyFeatures(n: number): Features {
  return { scalars: new Float64Array(n), vectors: new Float64Array(n * 3) }
}

/** ρ(g) acts as identity on scalars and R on vectors. */
export function applyRep(g: SE3, f: Features): Features {
  const n = f.scalars.length
  const out = emptyFeatures(n)
  out.scalars.set(f.scalars)
  for (let i = 0; i < n; i++) {
    const v: Vec3 = [f.vectors[i * 3], f.vectors[i * 3 + 1], f.vectors[i * 3 + 2]]
    const rv = applyR(g.R, v)
    out.vectors[i * 3] = rv[0]
    out.vectors[i * 3 + 1] = rv[1]
    out.vectors[i * 3 + 2] = rv[2]
  }
  return out
}

/** ε = ‖f(g·x) − ρ(g) f(x)‖ (combined scalar + vector Frobenius). */
export function residualNorm(fGx: Features, rhoFx: Features): number {
  const n = fGx.scalars.length
  let s = 0
  for (let i = 0; i < n; i++) {
    const ds = fGx.scalars[i] - rhoFx.scalars[i]
    s += ds * ds
  }
  for (let i = 0; i < fGx.vectors.length; i++) {
    const dv = fGx.vectors[i] - rhoFx.vectors[i]
    s += dv * dv
  }
  return Math.sqrt(s)
}

/** Per-point residual magnitude for heat tinting. */
export function perPointResidual(fGx: Features, rhoFx: Features): Float64Array {
  const n = fGx.scalars.length
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const ds = fGx.scalars[i] - rhoFx.scalars[i]
    const dx = fGx.vectors[i * 3] - rhoFx.vectors[i * 3]
    const dy = fGx.vectors[i * 3 + 1] - rhoFx.vectors[i * 3 + 1]
    const dz = fGx.vectors[i * 3 + 2] - rhoFx.vectors[i * 3 + 2]
    out[i] = Math.sqrt(ds * ds + dx * dx + dy * dy + dz * dz)
  }
  return out
}
