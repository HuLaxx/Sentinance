import type { Metadata } from "next";
import Script from "next/script";
import { Inter, Merienda, Bricolage_Grotesque } from "next/font/google";
import { StardustBackground } from "@/components/effects/stardust-background";
import { LoadingScreen } from "@/components/effects/loading-screen";
import { ProjectGuide } from "@/components/project-guide/project-guide";
import "./globals.css";
import "./loader.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const display = Bricolage_Grotesque({
  subsets: ["latin"],
  variable: "--font-display",
  axes: ["wdth", "opsz"],
  display: "swap",
});

const script = Merienda({
  subsets: ["latin"],
  weight: ["400", "600", "900"],
  variable: "--font-script",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Sentinance | Real-Time Crypto Market Intelligence",
  description: "Crypto and index prices streamed through Kafka into a durable ledger, historical forecasts, and an AI research agent that only explains a move when it can cite the evidence.",
  keywords: ["crypto", "market intelligence", "trading", "AI", "real-time analytics", "blockchain"],
  authors: [{ name: "Rahul Khanke", url: "https://portfolio.hulax.uk" }],
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
    images: [
      {
        url: "/icon.png",
        width: 512,
        height: 512,
        alt: "Sentinance Logo",
      },
    ],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${script.variable} ${display.variable} dark`} suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased bg-zinc-950 text-zinc-100`} suppressHydrationWarning>
        <Script id="remove-bis-attr" strategy="beforeInteractive">
          {`(function(){try{var nodes=document.querySelectorAll('[bis_skin_checked]');for(var i=0;i<nodes.length;i++){nodes[i].removeAttribute('bis_skin_checked');}}catch(e){}})();`}
        </Script>
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
        <ProjectGuide />
      </body>
    </html>
  );
}
