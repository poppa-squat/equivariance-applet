/**
 * Node sanity check: equivariant residual ≪ MLP residual under random SE(3).
 * Run: node scripts/sanity-residual.mjs
 */
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'
import { register } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

// Use tsx if available; otherwise transpile via vite-node — fall back to dynamic import of built logic.
// Simplest path: duplicate a tiny inline check by spawning vite-node / tsx.
const __dirname = dirname(fileURLToPath(import.meta.url))
const root = join(__dirname, '..')

async function main() {
  let tsx
  try {
    tsx = await import('tsx/esm/api')
  } catch {
    console.log('Installing tsx for sanity check…')
  }

  // Inline reimplementation enough for CI-less check via child process:
  const { spawnSync } = await import('node:child_process')
  const r = spawnSync(
    'npx',
    ['--yes', 'tsx', join(root, 'scripts/sanity-residual.ts')],
    { cwd: root, encoding: 'utf8', stdio: 'inherit' },
  )
  process.exit(r.status ?? 1)
}

main()
