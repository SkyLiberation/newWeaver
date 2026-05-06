import { Suspense } from 'react'
import { Chat } from '@/components/chat/Chat'

export default function Home() {
  return (
    <main className="h-screen">
      <Suspense fallback={null}>
        <Chat />
      </Suspense>
    </main>
  )
}
