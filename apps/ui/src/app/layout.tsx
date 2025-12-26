import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'LogMind AI',
  description: 'Local log observability platform with template mining and semantic search',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50">
        {children}
      </body>
    </html>
  );
}
