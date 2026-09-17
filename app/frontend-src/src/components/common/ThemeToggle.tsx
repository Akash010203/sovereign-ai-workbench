import React from 'react';
import { Sun, Moon } from 'lucide-react';
import { useUIStore } from '../../store/uiStore';

export default function ThemeToggle() {
  const { theme, toggleTheme } = useUIStore();
  return (
    <button onClick={toggleTheme} className="sidebar-bottom-btn">
      {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
      <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
    </button>
  );
}
