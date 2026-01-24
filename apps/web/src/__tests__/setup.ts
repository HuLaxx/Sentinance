import '@testing-library/jest-dom'

// Mock Next.js router
vi.mock('next/navigation', () => ({
    useRouter: () => ({
        push: vi.fn(),
        replace: vi.fn(),
        back: vi.fn(),
        forward: vi.fn(),
    }),
    usePathname: () => '/',
    useSearchParams: () => new URLSearchParams(),
}))

// Mock WebSocket
class MockWebSocket {
    onopen: (() => void) | null = null
    onmessage: ((event: MessageEvent) => void) | null = null
    onclose: (() => void) | null = null
    onerror: (() => void) | null = null
    readyState = 1

    send(_data: string) { }
    close() {
        this.readyState = 3
        if (this.onclose) this.onclose()
    }
}

global.WebSocket = MockWebSocket as any
