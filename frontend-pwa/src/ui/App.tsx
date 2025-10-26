import React, { useEffect, useMemo, useRef, useState } from 'react'
import { addDocument, searchTopK, DB } from '../data/vectorStore'
import { BrowserChat } from '../ml/chatBrowser'
import { backendStream } from '../ml/chatBackend'

type Engine = 'browser' | 'backend'

export default function App() {
  const [tab, setTab] = useState<'index'|'chat'|'search'>('index')
  const [engine, setEngine] = useState<Engine>('browser')
  const [busy, setBusy] = useState(false)
  const [text, setText] = useState('Paste or type a small document here.')
  const [query, setQuery] = useState('')
  const [top, setTop] = useState<{text:string, score:number}[]>([])
  const [messages, setMessages] = useState<{role:'user'|'assistant', content:string}[]>([])
  const chatRef = useRef<BrowserChat|null>(null)

  useEffect(() => {
    chatRef.current = new BrowserChat()
    chatRef.current.init().catch(console.error)
  }, [])

  const onIngest = async () => {
    setBusy(true)
    try {
      await addDocument({ id: 'doc-' + Date.now(), text, source: 'pasted' })
      alert('Indexed!')
    } finally {
      setBusy(false)
    }
  }

  const onSearch = async () => {
    setBusy(true)
    try {
      const results = await searchTopK(query, 5)
      setTop(results.map(r => ({ text: r.text, score: r.score })))
    } finally {
      setBusy(false)
    }
  }

  const onSend = async (input: string) => {
    setMessages(m => [...m, { role: 'user', content: input }])
    const rag = await searchTopK(input, 4)
    const context = rag.map(r => `- (${r.score.toFixed(2)}) ${r.text}`).join('\n')

    if (engine === 'browser') {
      const ans = await chatRef.current!.ask([
        { role: 'system', content: 'Answer concisely. Use the context when relevant.' },
        { role: 'user', content: `Context:\n${context}\n\nQuestion: ${input}` }
      ])
      setMessages(m => [...m, { role: 'assistant', content: ans }])
    } else {
      let acc = ''
      for await (const delta of backendStream('http://localhost:5000/chat', [
        { role: 'system', content: 'Answer concisely.' },
        { role: 'user', content: input }
      ])) {
        acc += delta
        setMessages(m => {
          const mm = [...m]
          mm[mm.length-1] = { role: 'assistant', content: acc }
          return mm
        })
      }
    }
  }

  return (
    <div style={{ fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif', padding: 16, maxWidth: 900, margin: '0 auto' }}>
      <h1>Offline AI — Browser + .NET</h1>
      <nav style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
        <button onClick={() => setTab('index')}>Index</button>
        <button onClick={() => setTab('search')}>Search</button>
        <button onClick={() => setTab('chat')}>Chat</button>
        <span style={{ marginLeft: 'auto' }}>
          Engine:
          <select value={engine} onChange={e => setEngine(e.target.value as Engine)}>
            <option value="browser">Browser (WebLLM)</option>
            <option value="backend">Backend (.NET)</option>
          </select>
        </span>
      </nav>

      {tab === 'index' && (
        <section>
          <textarea value={text} onChange={e => setText(e.target.value)} style={{ width: '100%', height: 200 }} />
          <div style={{ marginTop: 8 }}>
            <button disabled={busy} onClick={onIngest}>{busy ? 'Indexing…' : 'Index'}</button>
            <small style={{ marginLeft: 8 }}>Embeddings computed client-side and stored in IndexedDB.</small>
          </div>
        </section>
      )}

      {tab === 'search' && (
        <section>
          <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Ask something…" style={{ width: '100%' }} />
          <div style={{ marginTop: 8 }}>
            <button disabled={busy} onClick={onSearch}>Search</button>
          </div>
          <ul>
            {top.map((r,i) => <li key={i}><b>{r.score.toFixed(3)}</b> — {r.text}</li>)}
          </ul>
        </section>
      )}

      {tab === 'chat' && (
        <section>
          <ChatUI messages={messages} onSend={onSend} />
        </section>
      )}
    </div>
  )
}

function ChatUI({ messages, onSend }:{messages:{role:'user'|'assistant', content:string}[], onSend:(x:string)=>void}){
  const [input, setInput] = useState('What did I index?')
  return (
    <div>
      <div style={{ border:'1px solid #ddd', padding: 12, borderRadius: 8, height: 300, overflow: 'auto', background:'#fafafa' }}>
        {messages.map((m,i) => (
          <div key={i} style={{ marginBottom: 8 }}>
            <b>{m.role === 'user' ? 'You' : 'Assistant'}:</b> {m.content}
          </div>
        ))}
      </div>
      <div style={{ display:'flex', gap: 8, marginTop: 8 }}>
        <input value={input} onChange={e => setInput(e.target.value)} style={{ flex: 1 }} />
        <button onClick={() => onSend(input)}>Send</button>
      </div>
    </div>
  )
}