import Link from 'next/link';

export function SiteFooter() {
  return <footer className="atlas-footer">
    <span>Sentinance <small>by <a href="https://portfolio.hulax.uk">HuLaX</a></small></span>
    <div><a href="https://github.com/HuLaxx/Sentinance">GitHub</a>
      <a href="https://linkedin.com/in/rahulkhanke">LinkedIn</a>
      <a href="mailto:rahulkhanke02@gmail.com">Contact</a>
      <Link href="/privacy">Privacy</Link><Link href="/terms">Terms</Link><a href="#top">Back to top ↑</a>
    </div>
  </footer>;
}
