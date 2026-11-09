import { Controls } from './components/Controls'
import { Scene } from './components/Scene'
import { useDemoState } from './hooks/useDemoState'
import './App.css'

export default function App() {
  const demo = useDemoState()

  return (
    <div className="app">
      <div className="hero">
        <div className="hero-copy">
          <p className="brand">Equivariance</p>
          <h1>SE(3) on a living cloud</h1>
          <p className="lede">
            Rigid transforms should rotate vector features — not invent new ones.
            Watch an equivariant net keep the rule while a plain MLP breaks it.
          </p>
          <div className="cta-row">
            <button
              type="button"
              className="cta"
              onClick={() => demo.setAnimating(true)}
            >
              Drag to rotate the cloud
            </button>
            <button
              type="button"
              className="cta ghost-cta"
              onClick={demo.randomize}
            >
              Random SE(3)
            </button>
          </div>
        </div>
        <Scene demo={demo} />
        <Controls demo={demo} />
      </div>

      <section className="howto" aria-labelledby="howto-title">
        <h2 id="howto-title">How to read it</h2>
        <p className="howto-lede">
          Three claims, one residual — the whole demo fits in a sentence of math.
        </p>
        <ol>
          <li>
            <strong>SE(3)</strong> is the group of rigid motions: rotate by{' '}
            <em>R</em>, then translate by <em>t</em>. Points move as{' '}
            <code>x ↦ R x + t</code>.
          </li>
          <li>
            <strong>Equivariance</strong> says features follow the same rule:{' '}
            scalars stay put, vectors rotate with <em>R</em>. Formally{' '}
            <code>f(g·x) = ρ(g) f(x)</code>.
          </li>
          <li>
            <strong>The MLP fails</strong> because it flattens absolute
            coordinates — after a rotation its arrows wander. The message-passing
            toy only sees relative vectors, so its residual stays near zero.
          </li>
        </ol>
      </section>

      <footer className="foot">
        <span>Browser toys with fixed seeded weights · no training</span>
      </footer>
    </div>
  )
}
