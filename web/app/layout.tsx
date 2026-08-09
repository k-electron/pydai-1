import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'EDGAR Desk',
  description: 'A local-only research agent over SEC filings',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
