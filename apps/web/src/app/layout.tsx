import type { Metadata } from "next";
import Script from "next/script";
import { LoadingScreen } from "@/components/effects/loading-screen";
import { SiteHeader } from "@/components/site/site-header";
import { SiteFooter } from "@/components/site/site-footer";
import "./globals.css";
import "./atlas.css";

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
    <html lang="en" className="dark" suppressHydrationWarning>
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
