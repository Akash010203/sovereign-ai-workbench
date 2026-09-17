import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Server, Shield, Wifi } from 'lucide-react';
import ThemeToggle from '../components/common/ThemeToggle';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

export default function Settings() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      {/* Header */}
      <header className="flex items-center gap-3 px-6 py-4 border-b border-zinc-800">
        <button
          onClick={() => navigate('/')}
          className="text-zinc-400 hover:text-zinc-200 transition-colors"
        >
          <ArrowLeft size={18} />
        </button>
        <h1 className="text-base font-medium">Settings</h1>
      </header>

      {/* Content */}
      <div className="flex-1 max-w-2xl mx-auto w-full px-6 py-10 flex flex-col gap-8">
        {/* Appearance */}
        <section>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-4">
            Appearance
          </h2>
          <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4 flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-zinc-200">Theme</p>
              <p className="text-xs text-zinc-500">Toggle dark or light interface</p>
            </div>
            <ThemeToggle />
          </div>
        </section>

        {/* Backend */}
        <section>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-4">
            Backend Connection
          </h2>
          <div className="bg-zinc-900 rounded-xl border border-zinc-800 divide-y divide-zinc-800">
            <div className="p-4 flex items-center gap-3">
              <Server size={16} className="text-indigo-400 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium text-zinc-200">API Base URL</p>
                <p className="text-xs text-zinc-500 font-mono mt-0.5">{API_BASE}</p>
              </div>
            </div>
            <div className="p-4 flex items-center gap-3">
              <Wifi size={16} className="text-indigo-400 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium text-zinc-200">Local Only</p>
                <p className="text-xs text-zinc-500">
                  All requests go to your local machine. No data leaves your device.
                </p>
              </div>
            </div>
            <div className="p-4 flex items-center gap-3">
              <Shield size={16} className="text-indigo-400 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium text-zinc-200">Air-Gapped Operation</p>
                <p className="text-xs text-zinc-500">
                  SovereignAI is designed for offline, air-gapped environments.
                  Change VITE_API_BASE_URL in <code className="text-indigo-400">.env</code> to use a different host.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* About */}
        <section>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-4">
            About
          </h2>
          <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
            <p className="text-sm font-medium text-zinc-200">SovereignAI Workbench</p>
            <p className="text-xs text-zinc-500 mt-1">SIH 2026 · Problem Statement SIH26117</p>
            <p className="text-xs text-zinc-600 mt-3">
              Sovereign On-Premise Agentic AI Workbench — an air-gapped, fully local AI assistant for
              engineering, document analysis, and knowledge management. Built for Smart India Hackathon 2026.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
