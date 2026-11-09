import {
  applyRep,
  identitySE3,
  randomSE3,
  residualNorm,
  transformCloud,
} from '../src/se3/math'
import { moleculePoints } from '../src/se3/molecule'
import { forwardEquivariant, forwardMlp } from '../src/se3/models'
import { mulberry32 } from '../src/se3/rng'

const points = moleculePoints()
const rng = mulberry32(42)

const id = identitySE3()
const fxEq = forwardEquivariant(points)
const fxMlp = forwardMlp(points)
const fIdEq = forwardEquivariant(transformCloud(id, points))
const fIdMlp = forwardMlp(transformCloud(id, points))
const epsIdEq = residualNorm(fIdEq, applyRep(id, fxEq))
const epsIdMlp = residualNorm(fIdMlp, applyRep(id, fxMlp))

console.log('Identity residual:')
console.log(`  Equivariant: ${epsIdEq.toExponential(3)}`)
console.log(`  MLP:         ${epsIdMlp.toExponential(3)}`)

let maxEq = 0
let minMlp = Infinity
let sumEq = 0
let sumMlp = 0
const N = 24
for (let i = 0; i < N; i++) {
  const g = randomSE3(rng)
  const gx = transformCloud(g, points)
  const fGxEq = forwardEquivariant(gx)
  const fGxMlp = forwardMlp(gx)
  const eEq = residualNorm(fGxEq, applyRep(g, fxEq))
  const eMlp = residualNorm(fGxMlp, applyRep(g, fxMlp))
  maxEq = Math.max(maxEq, eEq)
  minMlp = Math.min(minMlp, eMlp)
  sumEq += eEq
  sumMlp += eMlp
  console.log(
    `g[${i}]  ε_eq=${eEq.toExponential(3)}  ε_mlp=${eMlp.toExponential(3)}  ratio=${(eMlp / Math.max(eEq, 1e-12)).toFixed(1)}×`,
  )
}

console.log('\nSummary:')
console.log(`  max ε_eq  = ${maxEq.toExponential(3)}`)
console.log(`  min ε_mlp = ${minMlp.toExponential(3)}`)
console.log(`  mean ε_eq = ${(sumEq / N).toExponential(3)}`)
console.log(`  mean ε_mlp= ${(sumMlp / N).toExponential(3)}`)

const ok = maxEq < 1e-6 && minMlp > 0.05 && sumMlp / N > 10 * (sumEq / N + 1e-12)
if (!ok) {
  console.error('\nFAIL: equivariant residual not clearly smaller than MLP.')
  process.exit(1)
}
console.log('\nPASS: equivariant residual near zero; MLP residual large.')
