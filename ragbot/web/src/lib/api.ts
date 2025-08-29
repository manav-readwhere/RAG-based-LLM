export async function* streamChat(query: string): AsyncGenerator<string> {
  const resp = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  })
  if (!resp.ok || !resp.body) {
    throw new Error('Failed to start chat stream')
  }
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    const chunk = decoder.decode(value)
    yield chunk
  }
}


