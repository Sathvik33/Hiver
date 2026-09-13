import { useState, useEffect } from 'react';

interface Evidence {
  conversation_id: string;
  similarity: number;
  customer_problem: string;
  resolution: string;
  quality: string;
}

interface AgentResponse {
  request_id: string;
  intent: string;
  intent_confidence: number;
  reply: string;
  decision: string;
  reason?: string;
  evidence: Evidence[];
  top_similarity: number;
  latency_ms: number;
}

export default function App() {
  const [tab, setTab] = useState<'playground' | 'evaluation' | 'failures'>('playground');
  const [message, setMessage] = useState('');
  const [context, setContext] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AgentResponse | null>(null);
  const [evalSummary, setEvalSummary] = useState<any>(null);
  const [failures, setFailures] = useState<any[]>([]);

  useEffect(() => {
    fetch('http://localhost:8000/api/evaluation/summary')
      .then(res => res.json())
      .then(data => setEvalSummary(data))
      .catch(() => {});

    fetch('http://localhost:8000/api/evaluation/results')
      .then(res => res.json())
      .then(data => setFailures(data.failures || []))
      .catch(() => {});
  }, []);

  const handleSend = async () => {
    if (!message.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/agent/respond', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, context: context || null })
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      alert('Error connecting to backend API. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const sampleQueries = [
    { label: "Delayed Package (Safe)", text: "@AmazonHelp Where is my order? It was supposed to be delivered yesterday." },
    { label: "Return Question (Safe)", text: "How do I print a return label for a pair of shoes that don't fit?" },
    { label: "Demand Human (Escalate)", text: "Stop sending automated bot replies! Let me speak to an actual person right now." },
    { label: "Account Hacked (Escalate)", text: "EMERGENCY: Someone changed my account email and charged $3,000 on gift cards!" },
    { label: "Ambiguous (Escalate)", text: "help" }
  ];

  return (
    <div className="app-container">
      <header>
        <div>
          <h1 style={{ fontSize: '1.6rem', fontWeight: 700 }}>Hiver AI Customer Support Agent</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Retrieval-First, Grounded AI Support with Conservative Escalation Guardrails
          </p>
        </div>
        <div>
          <span className="brand-badge">@AmazonHelp Support</span>
        </div>
      </header>

      <div className="tabs">
        <button className={`tab-btn ${tab === 'playground' ? 'active' : ''}`} onClick={() => setTab('playground')}>
          Agent Playground
        </button>
        <button className={`tab-btn ${tab === 'evaluation' ? 'active' : ''}`} onClick={() => setTab('evaluation')}>
          Evaluation Dashboard
        </button>
        <button className={`tab-btn ${tab === 'failures' ? 'active' : ''}`} onClick={() => setTab('failures')}>
          Failure Analysis
        </button>
      </div>

      {tab === 'playground' && (
        <div>
          <div className="quick-prompts">
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', alignSelf: 'center' }}>Test Presets:</span>
            {sampleQueries.map((q, idx) => (
              <button key={idx} className="quick-btn" onClick={() => setMessage(q.text)}>
                {q.label}
              </button>
            ))}
          </div>

          <div className="playground-grid">
            <div className="card">
              <h2 className="card-title">Customer Input</h2>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'block', marginBottom: 4 }}>
                Customer Tweet Message:
              </label>
              <textarea
                placeholder="Type incoming customer tweet or choose preset above..."
                value={message}
                onChange={e => setMessage(e.target.value)}
              />

              <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'block', marginBottom: 4 }}>
                Conversation Context (Optional):
              </label>
              <textarea
                style={{ minHeight: '60px' }}
                placeholder="Optional previous thread history..."
                value={context}
                onChange={e => setContext(e.target.value)}
              />

              <button className="primary-btn" onClick={handleSend} disabled={loading || !message.trim()}>
                {loading ? 'Processing via Qwen 2.5...' : 'Run Support Pipeline'}
              </button>
            </div>

            <div className="card">
              <div className="card-title">
                <span>Agent Decision & Response</span>
                {result && (
                  <span className={`badge ${result.decision === 'AUTO_HANDLE' ? 'badge-auto' : 'badge-escalate'}`}>
                    {result.decision}
                  </span>
                )}
              </div>

              {result ? (
                <div>
                  <div style={{ display: 'flex', gap: 16, marginBottom: 16, fontSize: '0.9rem' }}>
                    <div>
                      <span style={{ color: 'var(--text-secondary)' }}>Intent: </span>
                      <strong>{result.intent}</strong> ({Math.round(result.intent_confidence * 100)}%)
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-secondary)' }}>Top Similarity: </span>
                      <strong>{result.top_similarity}</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-secondary)' }}>Latency: </span>
                      <strong>{result.latency_ms}ms</strong>
                    </div>
                  </div>

                  {result.reason && (
                    <div style={{ background: 'var(--bg-tertiary)', padding: 10, borderRadius: 6, marginBottom: 14, fontSize: '0.85rem' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Reason: </span>
                      {result.reason}
                    </div>
                  )}

                  <div style={{ marginBottom: 16 }}>
                    <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'block', marginBottom: 4 }}>
                      Generated Reply:
                    </label>
                    <div style={{ background: 'var(--bg-primary)', padding: 14, borderRadius: 8, border: '1px solid var(--border-color)', fontSize: '0.95rem' }}>
                      {result.reply}
                    </div>
                  </div>

                  <div>
                    <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'block', marginBottom: 6 }}>
                      Retrieved Grounding Evidence ({result.evidence?.length || 0}):
                    </label>
                    {result.evidence?.map((ev, i) => (
                      <div key={i} className="evidence-box">
                        <div><strong>[{ev.conversation_id}] Sim: {ev.similarity} ({ev.quality})</strong></div>
                        <div style={{ color: 'var(--text-secondary)', marginTop: 4 }}>Problem: {ev.customer_problem}</div>
                        <div style={{ marginTop: 4 }}>Resolution: {ev.resolution}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: '40px 0' }}>
                  Enter a message and click 'Run Support Pipeline' to inspect intent classification, historical resolution retrieval, grounding validation, and escalation.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {tab === 'evaluation' && (
        <div>
          {evalSummary?.main_system ? (
            <div>
              <div className="metrics-grid">
                <div className="metric-card">
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Intent Macro-F1</div>
                  <div className="metric-value">{evalSummary.main_system.intent_metrics.macro_f1}</div>
                </div>
                <div className="metric-card">
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Retrieval Recall@5</div>
                  <div className="metric-value">{evalSummary.retrieval.recall_at_5}</div>
                </div>
                <div className="metric-card">
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Reply Quality (LLM Judge)</div>
                  <div className="metric-value">{evalSummary.main_system.reply_quality.avg_judge_overall} / 5.0</div>
                </div>
                <div className="metric-card">
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Unsafe Auto-Handle Rate</div>
                  <div className="metric-value" style={{ color: 'var(--accent-green)' }}>
                    {evalSummary.main_system.escalation_metrics.unsafe_auto_handle_rate}
                  </div>
                </div>
              </div>

              <div className="card" style={{ marginBottom: 20 }}>
                <h3 className="card-title">Comparative Baseline Benchmark</h3>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}>
                      <th style={{ padding: 10 }}>Model / Pipeline</th>
                      <th style={{ padding: 10 }}>Intent Macro-F1</th>
                      <th style={{ padding: 10 }}>Retrieval Recall@5</th>
                      <th style={{ padding: 10 }}>Unsafe Auto-Handle Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: 10 }}>Majority Class Baseline</td>
                      <td style={{ padding: 10 }}>{evalSummary.baselines.majority_class.intent_macro_f1}</td>
                      <td style={{ padding: 10 }}>N/A</td>
                      <td style={{ padding: 10 }}>{evalSummary.baselines.majority_class.unsafe_auto_handle_rate}</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: 10 }}>TF-IDF + Logistic Regression</td>
                      <td style={{ padding: 10 }}>{evalSummary.baselines.tfidf_logistic.intent_macro_f1}</td>
                      <td style={{ padding: 10 }}>{evalSummary.retrieval.recall_at_5}</td>
                      <td style={{ padding: 10 }}>{evalSummary.baselines.tfidf_logistic.unsafe_auto_handle_rate}</td>
                    </tr>
                    <tr style={{ fontWeight: 'bold', color: 'var(--accent-blue)' }}>
                      <td style={{ padding: 10 }}>Main System (Qwen 2.5 + FAISS + Guardrails)</td>
                      <td style={{ padding: 10 }}>{evalSummary.main_system.intent_metrics.macro_f1}</td>
                      <td style={{ padding: 10 }}>{evalSummary.retrieval.recall_at_5}</td>
                      <td style={{ padding: 10 }}>{evalSummary.main_system.escalation_metrics.unsafe_auto_handle_rate}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="card">
                <h3 className="card-title">Human-Judge Agreement Metrics (Audited Sample)</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, textAlign: 'center' }}>
                  <div style={{ background: 'var(--bg-tertiary)', padding: 12, borderRadius: 6 }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Pearson Correlation</div>
                    <div style={{ fontSize: '1.3rem', fontWeight: 600 }}>{evalSummary.main_system.reply_quality.agreement.pearson_correlation}</div>
                  </div>
                  <div style={{ background: 'var(--bg-tertiary)', padding: 12, borderRadius: 6 }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Spearman Rank</div>
                    <div style={{ fontSize: '1.3rem', fontWeight: 600 }}>{evalSummary.main_system.reply_quality.agreement.spearman_correlation}</div>
                  </div>
                  <div style={{ background: 'var(--bg-tertiary)', padding: 12, borderRadius: 6 }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Within ±1 Point</div>
                    <div style={{ fontSize: '1.3rem', fontWeight: 600 }}>
                      {Math.round(evalSummary.main_system.reply_quality.agreement.within_one_point_agreement_rate * 100)}%
                    </div>
                  </div>
                  <div style={{ background: 'var(--bg-tertiary)', padding: 12, borderRadius: 6 }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Cohen's Kappa</div>
                    <div style={{ fontSize: '1.3rem', fontWeight: 600 }}>{evalSummary.main_system.reply_quality.agreement.cohen_kappa}</div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: 40 }}>
              Evaluation results loading or benchmark is executing in the background...
            </div>
          )}
        </div>
      )}

      {tab === 'failures' && (
        <div>
          <h2 style={{ fontSize: '1.3rem', marginBottom: 16 }}>Root-Cause Failure Analysis</h2>
          {failures.length > 0 ? (
            failures.map((f, i) => (
              <div key={i} className="card" style={{ marginBottom: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <strong>Failure #{i + 1}: {f.type}</strong>
                  <span className="badge" style={{ background: 'rgba(210, 153, 34, 0.2)', color: 'var(--accent-amber)' }}>
                    {f.severity}
                  </span>
                </div>
                <div style={{ fontSize: '0.9rem', marginBottom: 6 }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Customer Input: </span> "{f.input}"
                </div>
                <div style={{ fontSize: '0.88rem', color: 'var(--accent-red)', marginBottom: 4 }}>
                  Why It Failed: {f.why_it_failed}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                  Hypothesis: {f.hypothesis}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--accent-green)' }}>
                  Potential Mitigation: {f.potential_fix}
                </div>
              </div>
            ))
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: 40 }}>
              No failure cases available yet.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
