import { test, expect } from '@playwright/test';

test('opens the market terminal from the home page', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/Sentinance/);
  await page.getByRole('link', { name: 'Open market terminal' }).click();
  await expect(page).toHaveURL(/\/demo$/);
  await expect(page.getByRole('heading', { name: 'Market Dashboard' })).toBeVisible();
});

test('opens the architecture from mobile navigation', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto('/');
  await page.getByRole('button', { name: 'Open navigation' }).click();
  await page.getByRole('banner').getByRole('link', { name: 'Architecture' }).click();
  await expect(page.getByRole('heading', { name: /Event-Driven\s*Microservices/ })).toBeVisible();
});
