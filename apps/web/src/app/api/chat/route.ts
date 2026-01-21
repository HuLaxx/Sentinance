export const runtime = "edge";

export async function POST(req: Request) {
    let body: { messages?: Array<{ role?: string; content?: string }> } | null = null;
    try {
        body = await req.json();
    } catch {
        return new Response("Invalid JSON payload", { status: 400 });
    }

    const messages = Array.isArray(body?.messages) ? body?.messages : [];
    const lastMessage = messages[messages.length - 1];
    if (!lastMessage?.content) {
        return new Response("No message content provided", { status: 400 });
    }

    // Call Python Backend
    const apiBaseUrl =
        process.env.API_BASE_URL ||
        process.env.NEXT_PUBLIC_API_URL ||
        "http://localhost:8000";
    const response = await fetch(`${apiBaseUrl}/api/chat`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            message: lastMessage.content,
            history: messages
                .slice(0, -1)
                .filter((message) => message?.content && message?.role)
                .map((message) => ({
                    role: message.role,
                    content: message.content,
                })),
            use_agent: true,
        }),
    });

    if (!response.ok) {
        return new Response("Backend Error", { status: 502 });
    }

    const data = await response.json();

    // Create a text encoder for streaming
    const encoder = new TextEncoder();

    // Create a stream from the response (since backend isn't streaming yet)
    const stream = new ReadableStream({
        start(controller) {
            const text = data?.content || "No analysis generated.";
            controller.enqueue(encoder.encode(text));
            controller.close();
        },
    });

    return new Response(stream, {
        headers: {
            "Content-Type": "text/plain; charset=utf-8",
        },
    });
}
