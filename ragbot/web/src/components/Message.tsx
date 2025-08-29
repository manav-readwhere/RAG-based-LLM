import React from 'react'

type Props = {
  role: 'user' | 'assistant'
  content: string
}

export function Message({ role, content }: Props) {
  const isAssistant = role === 'assistant'
  return (
    <div className={`p-3 my-2 bubble ${isAssistant ? 'bubble-assistant' : ''}`}>
      <div className="text-xs text-neon-cyan/80 mb-1">{isAssistant ? 'assistant' : 'you'}</div>
      <div className="whitespace-pre-wrap leading-relaxed">{content}</div>
    </div>
  )
}


