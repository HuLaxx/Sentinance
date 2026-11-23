import { expect, test } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";

test("real stored fixture and forecast evidence; tampered report fails closed", async ({ page }) => {
  const reportPath = process.env.P3_BROWSER_REPORT_PATH;
  if (!reportPath) throw new Error("Missing dedicated disposable report path");
  const digest = (await readFile(reportPath.replace(/\.json$/, ".sha256"), "utf8")).trim();
  const original = await readFile(reportPath);

  await page.goto("/data/evidence");
  await expect(page.getByRole("heading", { name: "Stored market events" })).toBeVisible();
  await expect(page.getByText("Synthetic fixture", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("20 stored BTCUSDT event(s)", { exact: true })).toBeVisible();
  await expect(page.getByRole("cell", { name: "ridge_ar1", exact: true })).toBeVisible();
  await expect(page.getByText("sha256:" + digest, { exact: true })).toBeVisible();
  await expect(page.getByText(/Trace: retrieve → analyze → finalize/)).toBeVisible();
  await page.screenshot({ path: "test-results/evidence-ready.png", fullPage: true });

  try {
    await writeFile(reportPath, Buffer.concat([original, Buffer.from("tampered")]));
    await page.reload();
    await expect(page.getByText("No verified forecast report configured.", { exact: true })).toBeVisible();
    await expect(page.getByRole("cell", { name: "ridge_ar1", exact: true })).toHaveCount(0);
    await expect(page.getByText("Synthetic fixture", { exact: true }).first()).toBeVisible();
    await page.screenshot({ path: "test-results/evidence-tamper.png", fullPage: true });
  } finally {
    await writeFile(reportPath, original);
  }
});
