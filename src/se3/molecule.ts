import type { Vec3 } from './math'

export interface Atom {
  position: Vec3
  /** Display radius */
  radius: number
  /** CSS-ish hex for atom color */
  color: string
  element: string
}

/**
 * Small asymmetric “molecule” so orientation is visually obvious —
 * vaguely like a chiral organometallic toy, not a real structure.
 */
export const MOLECULE: Atom[] = [
  { position: [0.0, 0.0, 0.0], radius: 0.28, color: '#5ec8c0', element: 'C' },
  { position: [1.15, 0.18, 0.12], radius: 0.22, color: '#e8e4dc', element: 'H' },
  { position: [-0.55, 1.05, 0.35], radius: 0.24, color: '#7eb8e8', element: 'N' },
  { position: [-0.72, -0.85, 0.55], radius: 0.26, color: '#e07a5f', element: 'O' },
  { position: [0.35, -0.25, -1.15], radius: 0.32, color: '#c9a227', element: 'S' },
  { position: [1.45, 1.05, -0.55], radius: 0.2, color: '#e8e4dc', element: 'H' },
  { position: [-1.55, 0.95, -0.25], radius: 0.22, color: '#8fd4a8', element: 'F' },
  { position: [0.85, -1.25, 0.15], radius: 0.2, color: '#e8e4dc', element: 'H' },
]

export function moleculePoints(): Vec3[] {
  return MOLECULE.map((a) => [...a.position] as Vec3)
}

export function centerOfMass(points: Vec3[]): Vec3 {
  const n = points.length || 1
  let x = 0
  let y = 0
  let z = 0
  for (const p of points) {
    x += p[0]
    y += p[1]
    z += p[2]
  }
  return [x / n, y / n, z / n]
}
