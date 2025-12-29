/**
 * API Route for ML Chat - handles long-running LLM requests
 * This bypasses Next.js rewrite proxy timeout limitations
 */

import { NextRequest, NextResponse } from 'next/server';

// Set maximum duration for this route (Vercel/Edge compatible)
export const maxDuration = 300; // 5 minutes

// Backend API URL
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Forward request to FastAPI backend with extended timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 400000); // 400 seconds
    
    const response = await fetch(`${BACKEND_URL}/ml/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        { error: `Backend error: ${response.status} - ${errorText}` },
        { status: response.status }
      );
    }
    
    const data = await response.json();
    return NextResponse.json(data);
    
  } catch (error) {
    console.error('ML Chat API error:', error);
    
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        return NextResponse.json(
          { error: 'Request timed out. The LLM is taking too long to respond.' },
          { status: 504 }
        );
      }
      return NextResponse.json(
        { error: error.message },
        { status: 500 }
      );
    }
    
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
