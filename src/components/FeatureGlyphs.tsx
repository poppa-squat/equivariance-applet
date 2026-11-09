import { useMemo } from 'react'
import * as THREE from 'three'
import type { Features, Vec3 } from '../se3/math'

function Arrow({
  origin,
  dir,
  color,
  opacity = 1,
}: {
  origin: Vec3
  dir: Vec3
  color: string
  opacity?: number
}) {
  const { quat, len } = useMemo(() => {
    const v = new THREE.Vector3(dir[0], dir[1], dir[2])
    const len = v.length()
    const quat = new THREE.Quaternion()
    if (len > 1e-6) {
      quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), v.clone().normalize())
    }
    return { quat, len }
  }, [dir])

  if (len < 1e-4) return null
  const shaft = Math.max(0.02, len * 0.72)
  const head = Math.max(0.04, len * 0.28)

  return (
    <group position={origin} quaternion={quat}>
      <mesh position={[0, shaft / 2, 0]}>
        <cylinderGeometry args={[0.025, 0.025, shaft, 8]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={opacity}
          emissive={color}
          emissiveIntensity={0.35}
          roughness={0.4}
        />
      </mesh>
      <mesh position={[0, shaft + head / 2, 0]}>
        <coneGeometry args={[0.06, head, 10]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={opacity}
          emissive={color}
          emissiveIntensity={0.45}
          roughness={0.35}
        />
      </mesh>
    </group>
  )
}

function ScalarRing({
  origin,
  value,
  color,
}: {
  origin: Vec3
  value: number
  color: string
}) {
  const r = 0.12 + Math.min(0.35, Math.abs(value) * 0.18)
  return (
    <mesh position={origin} rotation={[Math.PI / 2, 0, 0]}>
      <torusGeometry args={[r, 0.018, 8, 24]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.25 + Math.min(0.5, Math.abs(value) * 0.2)}
        transparent
        opacity={0.85}
        roughness={0.5}
      />
    </mesh>
  )
}

export function FeatureGlyphs({
  points,
  features,
  color,
  opacity = 1,
  showScalars = true,
}: {
  points: Vec3[]
  features: Features
  color: string
  opacity?: number
  showScalars?: boolean
}) {
  return (
    <group>
      {points.map((p, i) => {
        const dir: Vec3 = [
          features.vectors[i * 3],
          features.vectors[i * 3 + 1],
          features.vectors[i * 3 + 2],
        ]
        return (
          <group key={i}>
            <Arrow origin={p} dir={dir} color={color} opacity={opacity} />
            {showScalars && (
              <ScalarRing origin={p} value={features.scalars[i]} color={color} />
            )}
          </group>
        )
      })}
    </group>
  )
}

/** Check mode: show ρ(g)f(x) vs f(g·x) with residual heat on atoms optional. */
export function CheckOverlay({
  points,
  predicted,
  actual,
  predColor,
  actualColor,
}: {
  points: Vec3[]
  predicted: Features
  actual: Features
  predColor: string
  actualColor: string
}) {
  return (
    <group>
      <FeatureGlyphs
        points={points}
        features={predicted}
        color={predColor}
        opacity={0.45}
        showScalars={false}
      />
      <FeatureGlyphs
        points={points}
        features={actual}
        color={actualColor}
        opacity={1}
        showScalars
      />
    </group>
  )
}
