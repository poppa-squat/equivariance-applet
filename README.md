# Equivariance — SE(3) demo

Interactive browser applet that applies rigid SE(3) transforms to an asymmetric 3D point cloud and compares:

1. a **naive MLP** on flattened coordinates, and
2. an **SE(3)-equivariant** message-passing toy on relative vectors.

Live residual \(\varepsilon = \|f(g\cdot x) - \rho(g) f(x)\|\) stays near zero for the equivariant model and grows with \(g\) for the MLP.

## Run

```bash
npm install
npm run dev
```

Production build:

```bash
npm run build
npm run preview
```

Sanity check (equivariant residual ≪ MLP):

```bash
npm run sanity
```

## Stack

Vite · React 19 · TypeScript · React Three Fiber / Three.js · plain typed-array models (fixed seed).
