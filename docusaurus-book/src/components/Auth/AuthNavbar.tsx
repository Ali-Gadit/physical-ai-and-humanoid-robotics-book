import React from 'react';
import { authClient } from '../../lib/auth-client';
import { useHistory } from '@docusaurus/router';
import Link from '@docusaurus/Link';

export default function AuthNavbar() {
  const { data: session, isPending } = authClient.useSession();
  const history = useHistory();

  const handleSignOut = async () => {
    await authClient.signOut();
    history.push('/');
    window.location.reload(); 
  };

  if (isPending) return (
      <div className="navbar__item">Loading...</div>
  );

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
