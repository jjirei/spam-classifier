'use client';

import { useState } from 'react';

export default function Home() {
  const [message, setMessage] = useState('');
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });

      const data = await res.json();
      setResult(data.prediction === 1 ? 'Spam' : 'Not Spam');
    } catch (err) {
      console.error(err);
      setResult('Error contacting server');
    }

    setLoading(false);
  };

  return (
    <main style={{ padding: '2rem' }}>
      <h1>Spam Classifier</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          rows={4}
          cols={50}
          placeholder="Enter your message here"
        />
        <br />
        <button type="submit" disabled={loading}>
          {loading ? 'Classifying...' : 'Check Message'}
        </button>
      </form>
      {result && <p><strong>Result:</strong> {result}</p>}
    </main>
  );
}
