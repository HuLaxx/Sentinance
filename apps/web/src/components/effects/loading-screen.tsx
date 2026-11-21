'use client';

import { useEffect, useState } from 'react';

export function SignalLoader() {
  return <div className="atlas-loader" role="status">
    <div className="atlas-wave" aria-hidden="true">{[12, 28, 48, 72, 40, 22, 54, 36, 14].map((height, i) =>
      <i key={i} style={{height, animationDelay: `${i * 65}ms`}} />)}</div>
    <strong>Sentinance</strong><span>Opening the Signal Atlas</span>
  </div>;
}

export function LoadingScreen() {
  const [pending, setPending] = useState(true);
  useEffect(() => {
    if (document.readyState === 'complete') { setPending(false); return; }
    const loaded = () => setPending(false);
    window.addEventListener('load', loaded, {once: true});
    return () => window.removeEventListener('load', loaded);
  }, []);
  return pending ? <SignalLoader /> : null;
}
