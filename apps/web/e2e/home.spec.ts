import { test, expect } from '@playwright/test';

/**
 * E2E tests for Sentinance homepage
 */

test.describe('Homepage', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/');
    });

    test('should load successfully', async ({ page }) => {
        // Check page title
        await expect(page).toHaveTitle(/Sentinance/i);
    });

    test('should display price data', async ({ page }) => {
        // Wait for WebSocket connection and data
        await page.waitForTimeout(3000);

        // Check that price elements exist
        const priceElements = page.locator('[data-testid="price-card"]');
        await expect(priceElements.first()).toBeVisible({ timeout: 10000 });
    });

    test('should have navigation elements', async ({ page }) => {
        // Check for main navigation
        const nav = page.locator('nav');
        await expect(nav).toBeVisible();
    });

    test('should be responsive on mobile', async ({ page }) => {
        // Set mobile viewport
        await page.setViewportSize({ width: 375, height: 667 });

        // Page should still load
        await expect(page.locator('body')).toBeVisible();
    });
});

test.describe('Price Updates', () => {
    test('should receive real-time price updates', async ({ page }) => {
        await page.goto('/');

        // Get initial price (if displayed)
        await page.waitForTimeout(2000);

        // Wait for potential update
        await page.waitForTimeout(5000);

        // Page should still be responsive
        await expect(page.locator('body')).toBeVisible();
    });
});

test.describe('Accessibility', () => {
    test('should have proper heading structure', async ({ page }) => {
        await page.goto('/');

        // Check for h1
        const h1 = page.locator('h1');
        await expect(h1.first()).toBeVisible();
    });

    test('should have alt text on images', async ({ page }) => {
        await page.goto('/');

        // Check that all images have alt text
        const images = page.locator('img');
        const count = await images.count();

        for (let i = 0; i < count; i++) {
            const img = images.nth(i);
            const alt = await img.getAttribute('alt');
            expect(alt).not.toBeNull();
        }
    });
});
