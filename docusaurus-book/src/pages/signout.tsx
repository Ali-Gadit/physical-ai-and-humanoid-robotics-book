import React, { useEffect } from 'react';
import { useHistory } from '@docusaurus/router';
import { authClient } from '../lib/auth-client';
import Layout from '@theme/Layout';

export default function Signout() {
  const history = useHistory();

  useEffect(() => {
    authClient.signOut().then(() => {
      history.push('/signin');
    });
  }, [history]);

  return (
    <Layout title="Sign Out">
      <div className="container margin-vert--xl text--center">
        <h2>Signing out...</h2>
      </div>
    </Layout>
  );
}
