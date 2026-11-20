import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  testMatch: "evidence.spec.ts",
  workers: 1,
  retries: 0,
  timeout: 90000,
  reporter: [["list"], ["html", { open: "never", outputFolder: "evidence-report" }]],
  use: {
    ...devices["Desktop Chrome"],
    baseURL: "http://127.0.0.1:3000",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run dev -- --hostname 127.0.0.1",
    url: "http://127.0.0.1:3000/data/evidence",
    timeout: 180000,
    reuseExistingServer: false,
  },
});
