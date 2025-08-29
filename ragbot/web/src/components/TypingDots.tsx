import React from 'react'

export function TypingDots() {
  return (
    <span className="inline-flex gap-1 items-center text-neon-cyan">
      <span className="w-1.5 h-1.5 bg-neon-cyan animate-bounce rounded-full [animation-delay:-0.2s]"></span>
      <span className="w-1.5 h-1.5 bg-neon-cyan animate-bounce rounded-full [animation-delay:-0.1s]"></span>
      <span className="w-1.5 h-1.5 bg-neon-cyan animate-bounce rounded-full"></span>
    </span>
  )
}


