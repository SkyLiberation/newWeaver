'use client'

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { BookOpen, FileText, Loader2, RefreshCw, Trash2, Upload } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { getApiBaseUrl } from '@/lib/api'

interface KnowledgeBaseDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

interface RagDocument {
  id: string
  source?: string
  filename?: string
}

interface RagListResponse {
  total_chunks: number
  documents: RagDocument[]
}

interface DocumentGroup {
  source: string
  filename: string
  chunks: number
}

const ACCEPTED_EXTENSIONS = '.pdf,.doc,.docx,.txt,.md,.csv'

export function KnowledgeBaseDialog({ open, onOpenChange }: KnowledgeBaseDialogProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [documents, setDocuments] = useState<RagDocument[]>([])
  const [totalChunks, setTotalChunks] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [deletingSource, setDeletingSource] = useState<string | null>(null)

  const groupedDocuments = useMemo<DocumentGroup[]>(() => {
    const groups = new Map<string, DocumentGroup>()

    for (const item of documents) {
      const source = String(item.source || item.filename || item.id || '').trim()
      if (!source) continue

      const existing = groups.get(source)
      if (existing) {
        existing.chunks += 1
      } else {
        groups.set(source, {
          source,
          filename: String(item.filename || source).trim() || source,
          chunks: 1,
        })
      }
    }

    return Array.from(groups.values()).sort((a, b) => a.filename.localeCompare(b.filename))
  }, [documents])

  const loadDocuments = async () => {
    setIsLoading(true)
    try {
      const res = await fetch(`${getApiBaseUrl()}/api/documents/list?limit=500`, {
        cache: 'no-store',
      })
      if (!res.ok) {
        throw new Error(`Failed to load documents (${res.status})`)
      }
      const data = (await res.json()) as RagListResponse
      setDocuments(Array.isArray(data.documents) ? data.documents : [])
      setTotalChunks(typeof data.total_chunks === 'number' ? data.total_chunks : 0)
    } catch (error) {
      console.error('Failed to load RAG documents', error)
      toast.error('无法加载知识库文档')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (!open) return
    void loadDocuments()
  }, [open])

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleFilesSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    if (files.length === 0) return

    setIsUploading(true)
    try {
      for (const file of files) {
        const formData = new FormData()
        formData.append('file', file)

        const res = await fetch(`${getApiBaseUrl()}/api/documents/upload`, {
          method: 'POST',
          body: formData,
        })
        if (!res.ok) {
          let detail = `Upload failed (${res.status})`
          try {
            const data = await res.json()
            detail = data?.detail || detail
          } catch {
            // Ignore JSON parse errors.
          }
          throw new Error(detail)
        }

        const result = await res.json()
        toast.success(result?.message || `${file.name} 上传成功`)
      }

      await loadDocuments()
    } catch (error: any) {
      console.error('Failed to upload RAG document', error)
      toast.error(error?.message || '文档上传失败')
    } finally {
      setIsUploading(false)
      if (event.target) event.target.value = ''
    }
  }

  const handleDelete = async (source: string) => {
    setDeletingSource(source)
    try {
      const res = await fetch(`${getApiBaseUrl()}/api/documents/${encodeURIComponent(source)}`, {
        method: 'DELETE',
      })
      if (!res.ok) {
        let detail = `Delete failed (${res.status})`
        try {
          const data = await res.json()
          detail = data?.detail || detail
        } catch {
          // Ignore JSON parse errors.
        }
        throw new Error(detail)
      }

      toast.success('文档已删除')
      await loadDocuments()
    } catch (error: any) {
      console.error('Failed to delete RAG document', error)
      toast.error(error?.message || '文档删除失败')
    } finally {
      setDeletingSource(null)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[720px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-emerald-500" />
            知识库文档
          </DialogTitle>
          <DialogDescription>
            上传 PDF、Word、Markdown、TXT 或 CSV 文档，供“数据库”模式检索使用。
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-muted/20 px-4 py-3">
            <div className="text-sm text-muted-foreground">
              已收录 <span className="font-medium text-foreground">{groupedDocuments.length}</span> 份文档，
              共 <span className="font-medium text-foreground">{totalChunks}</span> 个分片
            </div>
            <div className="flex items-center gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPTED_EXTENSIONS}
                multiple
                className="hidden"
                onChange={handleFilesSelected}
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void loadDocuments()}
                disabled={isLoading || isUploading}
              >
                {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
                刷新
              </Button>
              <Button type="button" size="sm" onClick={handleUploadClick} disabled={isUploading}>
                {isUploading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Upload className="mr-2 h-4 w-4" />}
                上传文档
              </Button>
            </div>
          </div>

          <div className="rounded-xl border">
            {isLoading ? (
              <div className="flex min-h-40 items-center justify-center text-sm text-muted-foreground">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                正在加载知识库文档...
              </div>
            ) : groupedDocuments.length === 0 ? (
              <div className="flex min-h-40 flex-col items-center justify-center gap-2 px-6 text-center text-sm text-muted-foreground">
                <FileText className="h-8 w-8 text-muted-foreground/60" />
                <div>当前知识库还是空的。</div>
                <div>先上传文档后，数据库模式才会返回检索结果。</div>
              </div>
            ) : (
              <div className="max-h-[420px] overflow-y-auto">
                {groupedDocuments.map((doc) => (
                  <div key={doc.source} className="flex items-center justify-between gap-4 border-b px-4 py-3 last:border-b-0">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium">{doc.filename}</div>
                      <div className="truncate text-xs text-muted-foreground">source: {doc.source}</div>
                      <div className="mt-1 text-xs text-muted-foreground">{doc.chunks} 个分片</div>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="text-muted-foreground hover:text-destructive"
                      onClick={() => void handleDelete(doc.source)}
                      disabled={deletingSource === doc.source}
                    >
                      {deletingSource === doc.source ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="text-xs text-muted-foreground">
            支持格式：PDF、DOC、DOCX、TXT、MD、CSV。单文件上限 50MB。
          </div>
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
            关闭
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
