import { getApiBaseUrl } from '@/lib/api'

export interface PublicConfigModels {
  default: string
  options: string[]
}

export interface PublicConfigResponse {
  version: string
  defaults: {
    port: number
    primary_model: string
    reasoning_model: string
  }
  features: {
    mcp_enabled?: boolean
    rag_enabled?: boolean
    sandbox_mode?: string
    prometheus_enabled?: boolean
    tracing_enabled?: boolean
    [key: string]: any
  }
  streaming: Record<string, any>
  models?: PublicConfigModels
}

let publicConfigPromise: Promise<PublicConfigResponse | null> | null = null

export async function fetchPublicConfig(): Promise<PublicConfigResponse | null> {
  if (!publicConfigPromise) {
    publicConfigPromise = fetch(`${getApiBaseUrl()}/api/config/public`, { cache: 'no-store' })
      .then(async (res) => {
        if (!res.ok) return null
        return (await res.json()) as PublicConfigResponse
      })
      .catch(() => null)
  }
  return publicConfigPromise
}

