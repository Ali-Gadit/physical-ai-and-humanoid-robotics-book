import React from 'react';
import { authClient } from '../../lib/auth-client';
import { useHistory } from '@docusaurus/router';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';

interface AuthNavbarProps {
  mobile?: boolean;
  [key: string]: any;
}

export default function AuthNavbar({ mobile }: AuthNavbarProps) {
  const { data: session, isPending } = authClient.useSession();
  const history = useHistory();
  const homeUrl = useBaseUrl('/');

  const handleSignOut = async () => {
    await authClient.signOut();
    history.push(homeUrl);
    window.location.reload(); 
  };

  if (isPending) {
      if (mobile) return <li className="menu__list-item"><div className="menu__link">Loading...</div></li>;
      return <div className="navbar__item">Loading...</div>;
  }

  if (mobile) {
      if (session) {
          return (
            <li className="menu__list-item">
                <div className="menu__link">
                    Hi, {session.user.name}
                </div>
                <div style={{padding: '0 1rem', marginBottom: '1rem'}}>
                    <button className="button button--secondary button--block" onClick={handleSignOut}>
                    Sign Out
                    </button>
                </div>
            </li>
          );
      }
      return (
        <>
            <li className="menu__list-item">
                <Link className="menu__link" to="/signin">Sign In</Link>
            </li>
            <li className="menu__list-item">
                <Link className="menu__link" to="/signup">Sign Up</Link>
            </li>
        </>
      );
  }

  if (session) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <div className="navbar__item" style={{ marginRight: 0 }}>
            <span style={{ fontWeight: 'bold' }}>{session.user.name}</span>
        </div>
        <div className="navbar__item" style={{ paddingLeft: 0 }}>
            <button className="button button--sm button--secondary" onClick={handleSignOut}>
            Sign Out
            </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
      <div className="navbar__item">
        <Link className="button button--sm button--outline button--primary" to="/signin">Sign In</Link>
      </div>
      <div className="navbar__item" style={{ paddingLeft: 0 }}>
        <Link className="button button--sm button--primary" to="/signup">Sign Up</Link>
      </div>
    </div>
  );
}