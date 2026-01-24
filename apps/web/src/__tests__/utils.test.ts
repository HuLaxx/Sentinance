/**
 * Utility function tests
 */
import { describe, it, expect } from 'vitest'

// Utility functions to test
const formatPrice = (price: number): string => {
    return price.toLocaleString('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })
}

const formatPercentChange = (change: number): string => {
    const sign = change >= 0 ? '+' : ''
    return `${sign}${change.toFixed(2)}%`
}

const classifyChange = (change: number): 'positive' | 'negative' | 'neutral' => {
    if (change > 0) return 'positive'
    if (change < 0) return 'negative'
    return 'neutral'
}

describe('Price Formatting', () => {
    it('formats large prices correctly', () => {
        expect(formatPrice(95234.56)).toBe('$95,234.56')
    })

    it('formats small prices correctly', () => {
        expect(formatPrice(0.0015)).toBe('$0.00')
    })

    it('formats zero correctly', () => {
        expect(formatPrice(0)).toBe('$0.00')
    })
})

describe('Percent Change Formatting', () => {
    it('formats positive changes with + sign', () => {
        expect(formatPercentChange(2.45)).toBe('+2.45%')
    })

    it('formats negative changes with - sign', () => {
        expect(formatPercentChange(-3.21)).toBe('-3.21%')
    })

    it('formats zero as +0.00%', () => {
        expect(formatPercentChange(0)).toBe('+0.00%')
    })
})

describe('Change Classification', () => {
    it('classifies positive changes', () => {
        expect(classifyChange(1.5)).toBe('positive')
    })

    it('classifies negative changes', () => {
        expect(classifyChange(-0.5)).toBe('negative')
    })

    it('classifies zero as neutral', () => {
        expect(classifyChange(0)).toBe('neutral')
    })
})
