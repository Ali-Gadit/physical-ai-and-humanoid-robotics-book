import React, { useState, useEffect, ReactNode } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import LanguageToggle from './LanguageToggle';
import styles from './styles.module.css';
import { authClient } from '../../lib/auth-client';

interface BilingualChapterProps {
  children: ReactNode;
}

const PREF_KEY = 'bilingual_pref';

function BilingualChapterContent({ children }: BilingualChapterProps): JSX.Element {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isUrdu, setIsUrdu] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const session = await authClient.getSession();
        setIsAuthenticated(!!session.data);
        
        // Load preference only if authenticated
        if (session.data) {
             const savedPref = localStorage.getItem(PREF_KEY);
             if (savedPref === 'ur') {
                 setIsUrdu(true);
             }
        }
      } catch (error) {
        console.error("Auth check failed", error);
        setIsAuthenticated(false);
      } finally {
        setLoading(false);
      }
    };
    checkAuth();
  }, []);

  const handleToggle = () => {
    const newIsUrdu = !isUrdu;
    setIsUrdu(newIsUrdu);
    localStorage.setItem(PREF_KEY, newIsUrdu ? 'ur' : 'en');
  };

  if (loading) {
      return <div>Loading content...</div>;
  }
  
  // Cast children to an array to access them by index
  const childrenArray = React.Children.toArray(children);
  const englishContent = childrenArray[0];
  const urduContent = childrenArray[1];

  // If not authenticated, just show English content
  if (!isAuthenticated) {
    return <div className={styles.englishContent}>{englishContent}</div>;
  }

  return (
    <div>
      <LanguageToggle isUrdu={isUrdu} onToggle={handleToggle} />
      {isUrdu ? (
        <div className={styles.urduContent}>{urduContent}</div>
      ) : (
        <div className={styles.englishContent}>{englishContent}</div>
      )}
    </div>
  );
}

export default function BilingualChapter(props: BilingualChapterProps): JSX.Element {
  return (
    <BrowserOnly fallback={<div>Loading...</div>}>
      {() => <BilingualChapterContent {...props} />}
    </BrowserOnly>
  );
}
