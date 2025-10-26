export type ChatMessage = { role: 'system'|'user'|'assistant', content: string }

export async function* backendStream(url: string, messages: ChatMessage[]){
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages })
  })
  if (!resp.ok || !resp.body) throw new Error('HTTP error ' + resp.status)

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''

  while (true){
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })

    let idx
    while ((idx = buf.indexOf('\n\n')) !== -1){
      const raw = buf.slice(0, idx).trim()
      buf = buf.slice(idx + 2)
      if (raw.startsWith('data: ')){
        const data = raw.slice(6)
        if (data === 'end') return
        yield data
      }
    }
  }
}