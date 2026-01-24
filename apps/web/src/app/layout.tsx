import type { Metadata } from "next";
import { Inter, Merienda } from "next/font/google";
import { StardustBackground } from "@/components/effects/stardust-background";
import { LoadingScreen } from "@/components/effects/loading-screen";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const script = Merienda({
  subsets: ["latin"],
  weight: ["400", "600", "900"],
  variable: "--font-script",
  display: "swap",
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
    <html lang="en" className={`${script.variable} dark`} suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased bg-zinc-950 text-zinc-100`} suppressHydrationWarning>
        {/* Global animated gradient background */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
          <div className="absolute top-0 left-1/4 w-[700px] h-[700px] bg-gradient-to-r from-blue-500/15 via-cyan-500/15 to-blue-500/15 rounded-full blur-[150px] animate-pulse" style={{ animationDuration: '6s' }} />
          <div className="absolute bottom-0 right-1/4 w-[600px] h-[600px] bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-cyan-500/10 rounded-full blur-[150px] animate-pulse" style={{ animationDuration: '8s' }} />
          <div className="absolute top-1/3 right-1/3 w-[500px] h-[500px] bg-gradient-to-br from-cyan-500/10 via-transparent to-blue-500/10 rounded-full blur-[120px] animate-pulse" style={{ animationDuration: '10s' }} />
        </div>
        <StardustBackground />
        <LoadingScreen />
        <div className="noise" />
        {children}
      </body>
    </html>
  );
}
