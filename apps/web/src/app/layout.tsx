import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Sentinance | Real-Time Crypto Market Intelligence",
  description: "Enterprise-grade crypto market intelligence with agentic AI, real-time analytics, and ML-powered predictions.",
  keywords: ["crypto", "market intelligence", "trading", "AI", "real-time analytics", "blockchain"],
  authors: [{ name: "Sentinance Team" }],
  icons: {
    icon: [
      { url: "/icon.svg", type: "image/svg+xml" },
      { url: "/icon.png", type: "image/png" },
    ],
    apple: "/apple-icon.png",
  },
  openGraph: {
    title: "Sentinance | Real-Time Crypto Market Intelligence",
    description: "Enterprise-grade crypto market intelligence platform",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased bg-zinc-950 text-zinc-100`} suppressHydrationWarning>
        <div className="noise" />
        {children}
      </body>
    </html>
  );
}
