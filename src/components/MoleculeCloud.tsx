import { useMemo } from 'react'
import { Quaternion, Vector3 } from 'three'
import type { Vec3 } from '../se3/math'
import { MOLECULE } from '../se3/molecule'

function heatColor(t: number): string {
  const u = Math.min(1, Math.max(0, t))
  const r = Math.round(42 + u * (232 - 42))
  const g = Math.round(48 + u * (156 - 48))
  const b = Math.round(54 + u * (60 - 54))
  return `rgb(${r},${g},${b})`
}

export function MoleculeCloud({
  points,
  heat,
  heatMax = 1,
}: {
  points: Vec3[]
  heat?: Float64Array
  heatMax?: number
}) {
  const bonds = useMemo(() => {
    const edges: { mid: Vec3; quat: [number, number, number, number]; len: number }[] = []
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const a = points[i]
        const b = points[j]
        const dx = b[0] - a[0]
        const dy = b[1] - a[1]
        const dz = b[2] - a[2]
        const len = Math.hypot(dx, dy, dz)
        if (len >= 1.55 || len < 1e-6) continue
        const quat = new Quaternion().setFromUnitVectors(
          new Vector3(0, 1, 0),
          new Vector3(dx, dy, dz).normalize(),
        )
        edges.push({
          mid: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2],
          quat: [quat.x, quat.y, quat.z, quat.w],
          len,
        })
      }
    }
    return edges
  }, [points])

  return (
    <group>
      {points.map((p, i) => {
        const atom = MOLECULE[i]
        const color =
          heat && heatMax > 1e-8 ? heatColor(heat[i] / heatMax) : atom.color
        return (
          <mesh key={i} position={p} castShadow>
            <sphereGeometry args={[atom.radius, 28, 28]} />
            <meshStandardMaterial
              color={color}
              roughness={0.32}
              metalness={0.18}
              emissive={heat ? '#c9842a' : '#000000'}
              emissiveIntensity={
                heat ? Math.min(0.55, (heat[i] / heatMax) * 0.55) : 0
              }
            />
          </mesh>
        )
      })}
      {bonds.map((b, k) => (
        <mesh key={k} position={b.mid} quaternion={b.quat}>
          <cylinderGeometry args={[0.032, 0.032, b.len, 8]} />
          <meshStandardMaterial
            color="#7a848e"
            roughness={0.65}
            metalness={0.05}
            transparent
            opacity={0.5}
          />
        </mesh>
      ))}
    </group>
  )
}
