export const runtime = "nodejs";

const DEMO_PRICES = [
  { symbol: "BTCUSDT", price: 89877, change_24h: 0.09 },
  { symbol: "ETHUSDT", price: 2965, change_24h: 0.25 },
  { symbol: "SOLUSDT", price: 127.94, change_24h: -0.32 },
  { symbol: "XRPUSDT", price: 1.93, change_24h: 0.67 },
  { symbol: "BNBUSDT", price: 710, change_24h: 1.23 },
  { symbol: "ADAUSDT", price: 0.95, change_24h: 3.45 },
  { symbol: "DOGEUSDT", price: 0.38, change_24h: 8.92 },
  { symbol: "LINKUSDT", price: 28.5, change_24h: 2.15 },
  { symbol: "AVAXUSDT", price: 42.8, change_24h: 4.55 },
  { symbol: "DOTUSDT", price: 8.45, change_24h: -1.2 },
  { symbol: "MATICUSDT", price: 1.25, change_24h: 2.3 },
  { symbol: "LTCUSDT", price: 125.6, change_24h: 1.8 },
  { symbol: "^GSPC", price: 6915, change_24h: 0.03 },
  { symbol: "^DJI", price: 44156, change_24h: 0.12 },
  { symbol: "^IXIC", price: 19756, change_24h: -0.18 },
  { symbol: "^RUT", price: 2287, change_24h: 0.45 },
  { symbol: "^GSPTSE", price: 25120, change_24h: 0.22 },
  { symbol: "^BVSP", price: 128450, change_24h: -0.35 },
  { symbol: "^MXX", price: 56780, change_24h: 0.18 },
  { symbol: "^FTSE", price: 10143, change_24h: -0.07 },
  { symbol: "^GDAXI", price: 21520, change_24h: 0.35 },
  { symbol: "^FCHI", price: 7856, change_24h: 0.28 },
  { symbol: "^STOXX50E", price: 5089, change_24h: 0.42 },
  { symbol: "^AEX", price: 912, change_24h: 0.15 },
  { symbol: "^IBEX", price: 11890, change_24h: -0.22 },
  { symbol: "^SSMI", price: 12456, change_24h: 0.08 },
  { symbol: "^N225", price: 53847, change_24h: 0.29 },
  { symbol: "^NSEI", price: 25049, change_24h: -0.95 },
  { symbol: "^HSI", price: 19876, change_24h: 1.25 },
  { symbol: "000001.SS", price: 3256, change_24h: 0.45 },
  { symbol: "^KS11", price: 2534, change_24h: 0.67 },
  { symbol: "^TWII", price: 22890, change_24h: 0.38 },
  { symbol: "^AXJO", price: 8456, change_24h: 0.22 },
  { symbol: "^STI", price: 3678, change_24h: -0.12 },
  { symbol: "^BSESN", price: 82560, change_24h: -0.88 },
  { symbol: "^TA125", price: 2156, change_24h: 0.55 },
  { symbol: "^TASI", price: 12340, change_24h: 0.18 },
  { symbol: "^J203", price: 78650, change_24h: -0.42 },
];

export async function GET() {
  const apiBaseUrl =
    process.env.API_BASE_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://127.0.0.1:8000";

  try {
    const response = await fetch(`${apiBaseUrl}/api/prices`, {
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const data = await response.json();
    const isLive = typeof data?.isLive === "boolean" ? data.isLive : true;
    return Response.json({ ...data, isLive });
  } catch (e) {
    console.error("Prices proxy error:", e);
    return Response.json({ isLive: false, prices: DEMO_PRICES });
  }
}
