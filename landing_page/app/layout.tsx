import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Welcome to Our App',
  description: 'Landing page for our application',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
