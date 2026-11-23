import { test, expect } from '@playwright/test';

test('opens the terminal and drills into an asset', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/Sentinance/);
  await page.getByRole('link', { name: 'Open market terminal' }).click();
  await expect(page.getByRole('heading', { name: 'Market watchlist' })).toBeVisible();
  await page.getByRole('link', { name: /Bitcoin BTCUSDT/ }).click();
  await expect(page.getByRole('heading', { name: 'Bitcoin', exact: true })).toBeVisible();
});

test('opens the architecture from mobile navigation', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto('/');
  await page.getByRole('button', { name: 'Open navigation' }).click();
  await page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'Architecture' }).click();
  await expect(page.getByRole('heading', { name: 'Anatomy of a signal' })).toBeVisible();
});
