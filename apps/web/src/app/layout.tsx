import type { Metadata } from "next";
import Script from "next/script";
import { Hedvig_Letters_Serif, IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import { LoadingScreen } from "@/components/effects/loading-screen";
import { SiteHeader } from "@/components/site/site-header";
import { SiteFooter } from "@/components/site/site-footer";
import "./globals.css";
import "./atlas.css";

const hedvig = Hedvig_Letters_Serif({ subsets: ["latin"], weight: "400", variable: "--font-hedvig", display: "swap" });
const plexSans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-plex-sans", display: "swap" });
const plexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "600"], variable: "--font-plex-mono", display: "swap" });

export const metadata: Metadata = {
  title: "Sentinance | Real-Time Crypto Market Intelligence",
  description: "A crypto market research desk that traces every signal to its source: exchange ingestion through Kafka, PostgreSQL and Parquet, forecasting and LangGraph research agents.",
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
    description: "A crypto market research desk that traces every signal to its source",
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
    <html lang="en" className={`dark ${hedvig.variable} ${plexSans.variable} ${plexMono.variable}`} suppressHydrationWarning>
      <body id="top" className="antialiased" suppressHydrationWarning>
        <Script id="remove-bis-attr" strategy="beforeInteractive">
          {`(function(){try{var nodes=document.querySelectorAll('[bis_skin_checked]');for(var i=0;i<nodes.length;i++){nodes[i].removeAttribute('bis_skin_checked');}}catch(e){}})();`}
        </Script>
        <LoadingScreen />
        <SiteHeader />
        {children}
        <SiteFooter />
      </body>
    </html>
  );
}
