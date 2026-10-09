export class ApiConfigurationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ApiConfigurationError'
  }
}

export class ApiRequestError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiRequestError'
    this.status = status
  }
}

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  if (!apiBaseUrl) {
    throw new ApiConfigurationError('VITE_API_BASE_URL is not configured.')
  }
  if (!path.startsWith('/')) {
    throw new ApiConfigurationError('API paths must start with "/".')
  }

  const response = await fetch(`${apiBaseUrl.replace(/\/+$/, '')}${path}`, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...init?.headers,
    },
  })

  if (!response.ok) {
    throw new ApiRequestError(`API request failed with status ${response.status}.`, response.status)
  }

  return (await response.json()) as T
}
