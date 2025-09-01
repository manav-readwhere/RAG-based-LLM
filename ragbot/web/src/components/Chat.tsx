import React, { useRef, useState } from 'react'
import { streamChat, type ChatMessage } from '../lib/api'
import { Message } from './Message'
import { TypingDots } from './TypingDots'

export function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const assistantBuffer = useRef('')
  const endRef = useRef<HTMLDivElement | null>(null)

  const scrollToBottom = () => endRef.current?.scrollIntoView({ behavior: 'smooth' })

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!input.trim() || loading) return
    const q = input.trim()
    setInput('')
    const next = [...messages, { role: 'user', content: q }, { role: 'assistant', content: '' }]
    setMessages(next)
    setLoading(true)
    assistantBuffer.current = ''

    // Keep last N turns to send as context
    const CONTEXT_TURNS = 8
    const history = next.slice(Math.max(0, next.length - CONTEXT_TURNS * 2 - 1))

    try {
      for await (const token of streamChat(q, history)) {
        assistantBuffer.current += token
        setMessages(prev => {
          const copy = [...prev]
          copy[copy.length - 1] = { role: 'assistant', content: assistantBuffer.current }
          return copy
        })
        scrollToBottom()
      }
    } catch (e) {
      setMessages(prev => {
        const copy = [...prev]
        if (copy.length > 0 && copy[copy.length - 1].role === 'assistant') {
          copy[copy.length - 1] = { role: 'assistant', content: 'Error: failed to fetch response.' }
          return copy
        }
        return [...copy, { role: 'assistant', content: 'Error: failed to fetch response.' }]
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid grid-rows-[1fr_auto]">
      <div className="pr-2">
        {messages.length === 0 && (
          <div className="text-neon-cyan/80 mb-4">Ask a question about your ingested docs<span className="caret-blink"></span></div>
        )}
        {messages.map((m, i) => (
          <Message key={i} role={m.role} content={m.content} />
        ))}
        {loading && (
          <div className="p-2 text-sm"><TypingDots /></div>
        )}
        <div ref={endRef} />
      </div>
      <form onSubmit={onSubmit} className="mt-2 flex gap-2">
        <input
          className="flex-1 bg-zinc-900/80 text-green-200 border border-neon-cyan/30 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-neon-cyan/50"
          placeholder="Type your question..."
          value={input}
          onChange={e => setInput(e.target.value)}
        />
        <button
          type="submit"
          disabled={loading}
          className="px-4 py-2 border border-neon-green/50 text-neon-green hover:bg-neon-green/10 disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  )
}


