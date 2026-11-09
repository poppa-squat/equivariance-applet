import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  type Mat3,
  type SE3,
  type Vec3,
  applyRep,
  axisAngleToR,
  cloneSE3,
  composeSE3,
  matMul3,
  perPointResidual,
  randomSE3,
  residualNorm,
  transformCloud,
} from '../se3/math'
import { moleculePoints } from '../se3/molecule'
import { forwardEquivariant, forwardMlp } from '../se3/models'
import { mulberry32 } from '../se3/rng'

export type ViewMode = 'compare' | 'check'
export type ModelFocus = 'both' | 'equivariant' | 'mlp'

function eulerZYX(roll: number, pitch: number, yaw: number): Mat3 {
  // R = Rz(yaw) Ry(pitch) Rx(roll)
  const Rx = axisAngleToR([1, 0, 0], roll)
  const Ry = axisAngleToR([0, 1, 0], pitch)
  const Rz = axisAngleToR([0, 0, 1], yaw)
  return matMul3(Rz, matMul3(Ry, Rx))
}

export function useDemoState() {
  const basePoints = useMemo(() => moleculePoints(), [])
  const [euler, setEuler] = useState<Vec3>([0, 0, 0]) // roll, pitch, yaw
  const [t, setT] = useState<Vec3>([0, 0, 0])
  const [mode, setMode] = useState<ViewMode>('compare')
  const [focus, setFocus] = useState<ModelFocus>('both')
  const [animating, setAnimating] = useState(false)
  const animRef = useRef(0)
  const rngRef = useRef(mulberry32(0xc0ffee))

  const g: SE3 = useMemo(
    () => ({ R: eulerZYX(euler[0], euler[1], euler[2]), t }),
    [euler, t],
  )

  const transformed = useMemo(() => transformCloud(g, basePoints), [g, basePoints])

  const features = useMemo(() => {
    const fxEq = forwardEquivariant(basePoints)
    const fxMlp = forwardMlp(basePoints)
    const fGxEq = forwardEquivariant(transformed)
    const fGxMlp = forwardMlp(transformed)
    const rhoEq = applyRep(g, fxEq)
    const rhoMlp = applyRep(g, fxMlp)
    return {
      fxEq,
      fxMlp,
      fGxEq,
      fGxMlp,
      rhoEq,
      rhoMlp,
      epsEq: residualNorm(fGxEq, rhoEq),
      epsMlp: residualNorm(fGxMlp, rhoMlp),
      heatEq: perPointResidual(fGxEq, rhoEq),
      heatMlp: perPointResidual(fGxMlp, rhoMlp),
    }
  }, [basePoints, transformed, g])

  const reset = useCallback(() => {
    setEuler([0, 0, 0])
    setT([0, 0, 0])
  }, [])

  const randomize = useCallback(() => {
    const rg = randomSE3(rngRef.current)
    // Approximate: store translation; sample euler-ish from random angles
    setT(rg.t)
    setEuler([
      (rngRef.current() * 2 - 1) * Math.PI,
      (rngRef.current() * 2 - 1) * Math.PI * 0.5,
      (rngRef.current() * 2 - 1) * Math.PI,
    ])
  }, [])

  const setRoll = useCallback((v: number) => setEuler((e) => [v, e[1], e[2]]), [])
  const setPitch = useCallback((v: number) => setEuler((e) => [e[0], v, e[2]]), [])
  const setYaw = useCallback((v: number) => setEuler((e) => [e[0], e[1], v]), [])
  const setTranslation = useCallback((nt: Vec3) => setT([...nt] as Vec3), [])

  useEffect(() => {
    if (!animating) {
      cancelAnimationFrame(animRef.current)
      return
    }
    let last = performance.now()
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      setEuler((e) => [
        e[0] + dt * 0.25,
        e[1] + dt * 0.15,
        e[2] + dt * 0.55,
      ])
      setT([
        Math.sin(now / 1200) * 0.35,
        Math.cos(now / 1400) * 0.25,
        Math.sin(now / 1600) * 0.2,
      ])
      animRef.current = requestAnimationFrame(tick)
    }
    animRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animRef.current)
  }, [animating])

  return {
    g: cloneSE3(g),
    euler,
    basePoints,
    transformed,
    features,
    mode,
    setMode,
    focus,
    setFocus,
    animating,
    setAnimating,
    reset,
    randomize,
    setRoll,
    setPitch,
    setYaw,
    setTranslation,
    // kept for potential gizmo composition
    composeNudge: (dg: SE3) => {
      const next = composeSE3(dg, g)
      setT(next.t)
    },
  }
}

export type DemoState = ReturnType<typeof useDemoState>
