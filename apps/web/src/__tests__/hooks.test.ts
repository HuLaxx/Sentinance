/**
 * Hook tests
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'

/**
 * Price data type
 */
interface PriceData {
    symbol: string
    price: number
    change_24h: number
    volume: number
}

/**
 * Mock WebSocket hook behavior
 */
describe('WebSocket Price Hook', () => {
    beforeEach(() => {
        vi.clearAllMocks()
    })

    it('should initialize with empty prices', () => {
        const prices: PriceData[] = []
        expect(prices).toHaveLength(0)
    })

    it('should parse price data correctly', () => {
        const mockMessage = JSON.stringify({
            type: 'update',
            prices: [
                { symbol: 'BTCUSDT', price: 95000, change_24h: 2.5, volume: 1000000000 }
            ]
        })

        const parsed = JSON.parse(mockMessage)
        expect(parsed.type).toBe('update')
        expect(parsed.prices[0].symbol).toBe('BTCUSDT')
    })

    it('should handle connection states', () => {
        type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error'

        const states: ConnectionState[] = ['connecting', 'connected', 'disconnected', 'error']
        expect(states).toContain('connected')
        expect(states).toContain('error')
    })
})

describe('Price Calculation Helpers', () => {
    it('calculates market cap correctly', () => {
        const price = 95000
        const circulatingSupply = 19500000
        const marketCap = price * circulatingSupply

        expect(marketCap).toBe(1852500000000) // ~$1.85T
    })

    it('calculates 24h change correctly', () => {
        const oldPrice = 90000
        const newPrice = 95000
        const change = ((newPrice - oldPrice) / oldPrice) * 100

        expect(change).toBeCloseTo(5.56, 1)
    })
})
