export const runtime = "nodejs";

export async function GET(
    req: Request,
    { params }: { params: Promise<{ symbol: string }> }
) {
    const { symbol } = await params;
    const { searchParams } = new URL(req.url);
    const horizon = searchParams.get("horizon") || "24h";

    const apiBaseUrl =
        process.env.API_BASE_URL ||
        process.env.NEXT_PUBLIC_API_URL ||
        "http://127.0.0.1:8000";

    try {
        const response = await fetch(
            `${apiBaseUrl}/api/predict/${symbol}?horizon=${horizon}`,
            {
                headers: { "Content-Type": "application/json" },
                cache: "no-store",
            }
        );

        if (!response.ok) {
            return Response.json({ error: "Failed to fetch prediction" }, { status: response.status });
        }

        const data = await response.json();
        return Response.json(data);
    } catch (e) {
        console.error("Predict proxy error:", e);
        return Response.json({ error: "Backend unavailable" }, { status: 502 });
    }
}

