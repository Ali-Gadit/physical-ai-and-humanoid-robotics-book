import React from 'react';
import { Redirect, useLocation } from '@docusaurus/router';
import { authClient } from '../../lib/auth-client';

export default function AuthGuard({ children }: { children: React.ReactNode }) {
  const { data: session, isPending, error } = authClient.useSession();
  const location = useLocation();

  if (isPending) {
    return (
      <div className="container margin-vert--xl text--center">
        <div className="spinner-border" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
        <p>Verifying session...</p>
      </div>
    );
  }

  if (!session) {
    return <Redirect to={`/signin?redirect=${encodeURIComponent(location.pathname)}`} />;
  }

  return <>{children}</>;
}
