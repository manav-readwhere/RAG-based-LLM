import React from 'react'
import { Chat } from './components/Chat'
import '../tailwind.css'

export default function App() {
  return (
    <div className="min-h-screen bg-black text-neon-green">
      <header className="border-b border-cyan-500/30 p-4">
        <h1 className="text-2xl glow">RAGBot</h1>
        <p className="text-sm text-neon-cyan/80">Retrieval-Augmented Chatbot</p>
      </header>
      <main className="p-4 max-w-4xl mx-auto">
        <Chat />
      </main>
    </div>
  )
}


