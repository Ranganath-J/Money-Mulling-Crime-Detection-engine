export class ApiRequestError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiRequestError'
    this.status = status
  }
}

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() || '/api'

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  if (!path.startsWith('/')) {
    throw new Error('API paths must start with "/".')
  }

  const headers = new Headers(init?.headers)
  headers.set('Accept', 'application/json')
  const isFormData = typeof FormData !== 'undefined' && init?.body instanceof FormData
  if (init?.body && !isFormData && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const response = await fetch(`${apiBaseUrl.replace(/\/+$/, '')}${path}`, {
    ...init,
    headers,
  })

  if (!response.ok) {
    let message = `API request failed with status ${response.status}.`
    try {
      const payload = await response.json() as { error?: { message?: string } }
      if (payload.error?.message) message = payload.error.message
    } catch {
      // Keep the HTTP status message when the server response is not JSON.
    }
    throw new ApiRequestError(message, response.status)
  }

  return (await response.json()) as T
}
