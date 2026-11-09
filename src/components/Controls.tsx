import type { DemoState } from '../hooks/useDemoState'
import { vNorm } from '../se3/math'

export function Controls({ demo }: { demo: DemoState }) {
  const {
    g,
    euler,
    mode,
    setMode,
    focus,
    setFocus,
    animating,
    setAnimating,
    reset,
    randomize,
    setTranslation,
    setRoll,
    setPitch,
    setYaw,
  } = demo

  const tMag = vNorm(g.t)

  return (
    <aside className="panel readout">
      <header className="panel-head">
        <h2>Live residual</h2>
        <p className="panel-sub">ε = ‖f(g·x) − ρ(g) f(x)‖</p>
      </header>

      <div className="eps-stack" role="status" aria-live="polite">
        <EpsBar
          label="Equivariant"
          value={demo.features.epsEq}
          tone="teal"
          hint="should stay near zero"
          maxRef={Math.max(demo.features.epsMlp, demo.features.epsEq, 0.5)}
        />
        <EpsBar
          label="Naive MLP"
          value={demo.features.epsMlp}
          tone="coral"
          hint="grows with g"
          maxRef={Math.max(demo.features.epsMlp, demo.features.epsEq, 0.5)}
        />
      </div>

      <div className="seg" role="tablist" aria-label="Visualization mode">
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'compare'}
          className={mode === 'compare' ? 'on' : ''}
          onClick={() => setMode('compare')}
        >
          Compare
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'check'}
          className={mode === 'check' ? 'on' : ''}
          onClick={() => setMode('check')}
        >
          Check
        </button>
      </div>
      <p className="mode-hint">
        {mode === 'compare'
          ? 'Teal = equivariant features · Coral = MLP features after the same g.'
          : 'Dim = ρ(g)f(x) predicted · Bright = f(g·x). Atoms heat by residual.'}
      </p>

      <div className="seg subtle" role="tablist" aria-label="Model focus">
        {(['both', 'equivariant', 'mlp'] as const).map((k) => (
          <button
            key={k}
            type="button"
            role="tab"
            aria-selected={focus === k}
            className={focus === k ? 'on' : ''}
            onClick={() => setFocus(k)}
          >
            {k === 'both' ? 'Both' : k === 'equivariant' ? 'SE(3)' : 'MLP'}
          </button>
        ))}
      </div>

      <div className="sliders">
        <label>
          <span>Yaw</span>
          <input
            type="range"
            min={-Math.PI}
            max={Math.PI}
            step={0.01}
            value={euler[2]}
            onChange={(e) => setYaw(Number(e.target.value))}
          />
        </label>
        <label>
          <span>Pitch</span>
          <input
            type="range"
            min={-Math.PI / 2}
            max={Math.PI / 2}
            step={0.01}
            value={euler[1]}
            onChange={(e) => setPitch(Number(e.target.value))}
          />
        </label>
        <label>
          <span>Roll</span>
          <input
            type="range"
            min={-Math.PI}
            max={Math.PI}
            step={0.01}
            value={euler[0]}
            onChange={(e) => setRoll(Number(e.target.value))}
          />
        </label>
        <label>
          <span>Translate X</span>
          <input
            type="range"
            min={-1.5}
            max={1.5}
            step={0.01}
            value={g.t[0]}
            onChange={(e) =>
              setTranslation([Number(e.target.value), g.t[1], g.t[2]])
            }
          />
        </label>
        <label>
          <span>Translate Y</span>
          <input
            type="range"
            min={-1.5}
            max={1.5}
            step={0.01}
            value={g.t[1]}
            onChange={(e) =>
              setTranslation([g.t[0], Number(e.target.value), g.t[2]])
            }
          />
        </label>
        <label>
          <span>Translate Z</span>
          <input
            type="range"
            min={-1.5}
            max={1.5}
            step={0.01}
            value={g.t[2]}
            onChange={(e) =>
              setTranslation([g.t[0], g.t[1], Number(e.target.value)])
            }
          />
        </label>
      </div>

      <div className="actions">
        <button type="button" className="ghost" onClick={reset}>
          Reset
        </button>
        <button type="button" className="ghost" onClick={randomize}>
          Random SE(3)
        </button>
        <button
          type="button"
          className={animating ? 'ghost on' : 'ghost'}
          onClick={() => setAnimating(!animating)}
        >
          {animating ? 'Stop' : 'Animate g'}
        </button>
      </div>

      <p className="g-meta">‖t‖ = {tMag.toFixed(2)}</p>
    </aside>
  )
}

function EpsBar({
  label,
  value,
  tone,
  hint,
  maxRef,
}: {
  label: string
  value: number
  tone: 'teal' | 'coral'
  hint: string
  maxRef?: number
}) {
  const cap = maxRef ?? Math.max(value, 0.35)
  const pct = Math.min(100, (value / cap) * 100)
  return (
    <div className={`eps ${tone}`}>
      <div className="eps-top">
        <span>{label}</span>
        <strong>{value.toExponential(2)}</strong>
      </div>
      <div className="eps-track" aria-hidden>
        <div className="eps-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="eps-hint">{hint}</span>
    </div>
  )
}
