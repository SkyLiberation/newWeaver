'use client'

import { useState, useEffect, useCallback } from 'react'
import { ChatSession, Message } from '@/types/chat'
import { StorageService } from '@/lib/storage-service'

interface SaveHistoryOptions {
  threadId?: string | null
}

function buildSessionFingerprint(session: ChatSession): string {
  const messages = StorageService.getSessionMessages<Message>(session.id) || []
  const normalized = messages
    .map(m => `${m.role}:${String(m.content || '').trim()}`)
    .filter(Boolean)
    .slice(0, 6)
    .join('|')

  return [
    session.threadId || '',
    String(session.title || '').trim(),
    normalized
  ].join('::')
}

function normalizeHistory(sessions: ChatSession[]): ChatSession[] {
  const byThreadId = new Map<string, ChatSession>()
  const byFingerprint = new Map<string, ChatSession>()

  for (const rawSession of sessions) {
    const session: ChatSession = {
      ...rawSession,
      createdAt: rawSession.createdAt || Date.now(),
      updatedAt: rawSession.updatedAt || Date.now(),
      isPinned: rawSession.isPinned || false,
      tags: rawSession.tags || []
    }

    const threadId = (session.threadId || '').trim()
    if (threadId) {
      const existing = byThreadId.get(threadId)
      if (!existing || existing.updatedAt < session.updatedAt) {
        byThreadId.set(threadId, session)
      }
      continue
    }

    const fingerprint = buildSessionFingerprint(session)
    const existing = byFingerprint.get(fingerprint)
    if (!existing || existing.updatedAt < session.updatedAt) {
      byFingerprint.set(fingerprint, session)
    }
  }

  return [...byThreadId.values(), ...byFingerprint.values()].sort((a, b) => {
    if (a.isPinned && !b.isPinned) return -1
    if (!a.isPinned && b.isPinned) return 1
    return b.updatedAt - a.updatedAt
  })
}

export function useChatHistory() {
  const [history, setHistory] = useState<ChatSession[]>([])
  const [isHistoryLoading, setIsHistoryLoading] = useState(true)

  // Load History
  useEffect(() => {
    const loadHistory = () => {
      try {
        const savedHistory = StorageService.getHistory<ChatSession>()
        setHistory(normalizeHistory(savedHistory))
      } catch (e) {
        console.error('Failed to load history', e)
      } finally {
        setIsHistoryLoading(false)
      }
    }

    loadHistory()
  }, [])

  // Persist changes whenever history updates
  useEffect(() => {
    if (!isHistoryLoading) {
      StorageService.saveHistory(history)
    }
  }, [history, isHistoryLoading])

  const refreshHistory = useCallback(() => {
    const saved = StorageService.getHistory<ChatSession>()
    setHistory(normalizeHistory(saved))
  }, [])

  const saveToHistory = useCallback((
    messages: Message[],
    currentSessionId?: string,
    options?: SaveHistoryOptions
  ) => {
    if (messages.length === 0) return null

    const timestamp = Date.now()
    let sessionId = currentSessionId
    const nextThreadId = options?.threadId || undefined

    setHistory(prev => {
      const existingIndex = sessionId ? prev.findIndex(s => s.id === sessionId) : -1
      
      if (existingIndex !== -1) {
        // Update existing session
        const updatedHistory = [...prev]
        updatedHistory[existingIndex] = {
          ...updatedHistory[existingIndex],
          updatedAt: timestamp,
          threadId: nextThreadId ?? updatedHistory[existingIndex].threadId,
          // Update title if it's still the default "New Conversation" or generic
          // (Logic can be refined, here we just update timestamp primarily)
        }
        // Re-sort
        return updatedHistory.sort((a, b) => {
           if (a.isPinned && !b.isPinned) return -1
           if (!a.isPinned && b.isPinned) return 1
           return b.updatedAt - a.updatedAt
        })
      } else {
        // Create new session
        sessionId = sessionId || timestamp.toString()
        const firstUserMsg = messages.find(m => m.role === 'user')
        const title = firstUserMsg ? firstUserMsg.content.slice(0, 40) : 'New Conversation'
        
        const newSession: ChatSession = {
          id: sessionId,
          title,
          date: new Date(timestamp).toLocaleDateString(), // Keep legacy for display fallback
          createdAt: timestamp,
          updatedAt: timestamp,
          threadId: nextThreadId,
          isPinned: false,
          tags: []
        }
        return normalizeHistory([newSession, ...prev])
      }
    })

    // Save messages content
    if (sessionId) {
      StorageService.saveSessionMessages(sessionId, messages)
    }
    
    return sessionId
  }, [])

  const loadSession = useCallback((id: string): Message[] | null => {
    return StorageService.getSessionMessages(id)
  }, [])

  const deleteSession = useCallback((id: string) => {
    setHistory(prev => prev.filter(s => s.id !== id))
    StorageService.removeSessionMessages(id)
  }, [])

  const clearHistory = useCallback(() => {
    StorageService.clearAll()
    setHistory([])
  }, [])

  const togglePin = useCallback((id: string) => {
    setHistory(prev => {
      const mapped = prev.map(s => s.id === id ? { ...s, isPinned: !s.isPinned } : s)
      return mapped.sort((a, b) => {
          if (a.isPinned && !b.isPinned) return -1
          if (!a.isPinned && b.isPinned) return 1
          return b.updatedAt - a.updatedAt
      })
    })
  }, [])

  const renameSession = useCallback((id: string, newTitle: string) => {
    setHistory(prev => prev.map(s => s.id === id ? { ...s, title: newTitle } : s))
  }, [])

  const setSessionThreadId = useCallback((id: string, threadId: string | null) => {
    if (!id || !threadId) return
    setHistory(prev => {
      let changed = false
      const next = prev.map(s => {
        if (s.id !== id || s.threadId === threadId) return s
        changed = true
        return { ...s, threadId }
      })
      return changed ? next : prev
    })
  }, [])

  const updateTags = useCallback((id: string, tags: string[]) => {
    setHistory(prev => prev.map(s => s.id === id ? { ...s, tags } : s))
  }, [])

  return {
    history,
    isHistoryLoading,
    saveToHistory,
    loadSession,
    deleteSession,
    clearHistory,
    togglePin,
    renameSession,
    setSessionThreadId,
    updateTags,
    refreshHistory
  }
}
