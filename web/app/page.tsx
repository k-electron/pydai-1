'use client';

import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { useEffect, useMemo, useRef, useState } from 'react';

const SUGGESTIONS = [
  "How has NVIDIA's R&D spend as a share of revenue changed over five years?",
  'Compare gross margins for AAPL, MSFT and GOOGL in the most recent fiscal year.',
  'What does Intel say about competition in its latest risk factors?',
  'Which covered company spends the most on R&D relative to revenue?',
];

type Company = { ticker: string; name: string; cik: string | null };

export default function Page() {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  const transport = useMemo(() => new DefaultChatTransport({ api: '/api/chat' }), []);
  const { messages, sendMessage, status, error, addToolApprovalResponse } = useChat({
    transport,
  });

  useEffect(() => {
    fetch('/api/companies')
      .then((r) => r.json())
      .then(setCompanies)
      .catch(() => setCompanies([]));
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const busy = status === 'submitted' || status === 'streaming';

  function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    sendMessage({ text: trimmed });
    setInput('');
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">EDGAR Desk</div>
        <div className="brand-sub">Local models, SEC filings only</div>

        <div className="section-label">Try asking</div>
        {SUGGESTIONS.map((s) => (
          <button key={s} className="suggestion" onClick={() => submit(s)} disabled={busy}>
            {s}
          </button>
        ))}

        <div className="section-label">Covered ({companies.length})</div>
        <div className="ticker-grid">
          {companies.map((c) => (
            <span key={c.ticker} className="ticker" title={`${c.name} — CIK ${c.cik ?? '?'}`}>
              {c.ticker}
            </span>
          ))}
        </div>
      </aside>

      <main className="main">
        <div className="messages" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="empty">
              <p>
                Ask about reported financials or what these companies say in their 10-K
                filings. Numbers come from XBRL and are exact; narrative passages are
                retrieved from filing text.
              </p>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`msg ${message.role}`}>
              <div className="role">{message.role === 'user' ? 'You' : 'EDGAR Desk'}</div>
              {message.parts.map((part, index) => {
                if (part.type === 'text') {
                  return (
                    <div className="bubble" key={index}>
                      {part.text}
                    </div>
                  );
                }

                // Approval-gated tools arrive as a request the user has to answer
                // before the run continues.
                if (
                  part.type?.startsWith('tool-') &&
                  'state' in part &&
                  part.state === 'approval-requested'
                ) {
                  const toolName = part.type.replace('tool-', '');
                  const callId = (part as { toolCallId: string }).toolCallId;
                  return (
                    <div className="approval" key={index}>
                      <h4>Approval needed: {toolName}</h4>
                      <p>
                        The agent wants to run <code>{toolName}</code>. Nothing is written
                        until you approve.
                      </p>
                      <div className="approval-actions">
                        <button
                          className="primary"
                          onClick={() =>
                            addToolApprovalResponse({ id: callId, approved: true })
                          }
                        >
                          Approve
                        </button>
                        <button
                          className="secondary"
                          onClick={() =>
                            addToolApprovalResponse({
                              id: callId,
                              approved: false,
                              reason: 'Declined by reviewer',
                            })
                          }
                        >
                          Deny
                        </button>
                      </div>
                    </div>
                  );
                }

                if (part.type?.startsWith('tool-')) {
                  const toolName = part.type.replace('tool-', '');
                  const state = 'state' in part ? String(part.state) : '';
                  return (
                    <div className="tool" key={index}>
                      <span className="tool-name">{toolName}</span>
                      {state ? ` · ${state.replace('output-', '').replace('input-', '')}` : ''}
                    </div>
                  );
                }

                return null;
              })}
            </div>
          ))}

          {busy && <div className="msg"><div className="role">EDGAR Desk</div><div className="bubble">…</div></div>}
          {error && <div className="msg"><div className="bubble error">{error.message}</div></div>}
        </div>

        <div className="composer">
          <form
            className="composer-inner"
            onSubmit={(e) => {
              e.preventDefault();
              submit(input);
            }}
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about a covered company…"
              disabled={busy}
            />
            <button className="primary" type="submit" disabled={busy || !input.trim()}>
              Send
            </button>
          </form>
          <div className="status">
            Local inference — a multi-company question can take a minute or two.
          </div>
        </div>
      </main>
    </div>
  );
}
