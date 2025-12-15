import React from 'react';
import styles from './styles.module.css';

interface LanguageToggleProps {
  isUrdu: boolean;
  onToggle: () => void;
}

export default function LanguageToggle({ isUrdu, onToggle }: LanguageToggleProps): JSX.Element {
  return (
    <button className={styles.toggleButton} onClick={onToggle}>
      {isUrdu ? 'Translate to English' : 'Translate to Urdu'}
    </button>
  );
}
