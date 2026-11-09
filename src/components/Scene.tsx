import { Canvas } from '@react-three/fiber'
import { ContactShadows, Environment, OrbitControls } from '@react-three/drei'
import { Suspense } from 'react'
import type { DemoState } from '../hooks/useDemoState'
import { CheckOverlay, FeatureGlyphs } from './FeatureGlyphs'
import { MoleculeCloud } from './MoleculeCloud'

const TEAL = '#3dcfbf'
const AMBER = '#e8a54b'
const MLP = '#d4786a'

function SceneContent({ demo }: { demo: DemoState }) {
  const { transformed, features, mode, focus } = demo
  const heat =
    mode === 'check'
      ? focus === 'mlp'
        ? features.heatMlp
        : features.heatEq
      : undefined
  const heatMax = heat
    ? Math.max(1e-6, ...Array.from(heat))
    : 1

  const showEq = focus === 'both' || focus === 'equivariant'
  const showMlp = focus === 'both' || focus === 'mlp'

  return (
    <>
      <color attach="background" args={['#14181c']} />
      <ambientLight intensity={0.45} />
      <directionalLight position={[4, 7, 3]} intensity={1.15} castShadow />
      <directionalLight position={[-3, 2, -4]} intensity={0.35} color="#8ec8ff" />

      <group position={[0, 0.1, 0]}>
        <MoleculeCloud points={transformed} heat={heat} heatMax={heatMax} />

        {mode === 'compare' && showEq && (
          <FeatureGlyphs
            points={transformed}
            features={features.fGxEq}
            color={TEAL}
          />
        )}
        {mode === 'compare' && showMlp && (
          <FeatureGlyphs
            points={transformed}
            features={features.fGxMlp}
            color={MLP}
            opacity={focus === 'both' ? 0.75 : 1}
          />
        )}

        {mode === 'check' && showEq && (
          <CheckOverlay
            points={transformed}
            predicted={features.rhoEq}
            actual={features.fGxEq}
            predColor="#8fd4cc"
            actualColor={TEAL}
          />
        )}
        {mode === 'check' && showMlp && (
          <CheckOverlay
            points={transformed}
            predicted={features.rhoMlp}
            actual={features.fGxMlp}
            predColor="#e0b090"
            actualColor={AMBER}
          />
        )}
      </group>

      <ContactShadows
        position={[0, -1.85, 0]}
        opacity={0.45}
        scale={12}
        blur={2.5}
        far={4}
      />
      <OrbitControls
        makeDefault
        enablePan={false}
        minDistance={3.2}
        maxDistance={10}
        maxPolarAngle={Math.PI * 0.92}
      />
      <Environment preset="city" environmentIntensity={0.35} />
    </>
  )
}

export function Scene({ demo }: { demo: DemoState }) {
  return (
    <div className="scene-wrap">
      <Canvas
        shadows
        dpr={[1, 1.75]}
        camera={{ position: [3.6, 2.4, 4.2], fov: 42 }}
        gl={{ antialias: true, alpha: false }}
      >
        <Suspense fallback={null}>
          <SceneContent demo={demo} />
        </Suspense>
      </Canvas>
    </div>
  )
}
