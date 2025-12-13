import React from 'react';
import Layout from '@theme/Layout';
import SigninForm from '../components/Auth/SigninForm';

export default function Signin() {
  return (
    <Layout title="Signin" description="Sign in to your account">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--6 col--offset-3">
            <div className="card shadow--md">
              <div className="card__header">
                <h2>Welcome Back</h2>
                <p>Sign in to continue your learning journey</p>
              </div>
              <div className="card__body">
                <SigninForm />
              </div>
              <div className="card__footer text--center">
                 <p>Don't have an account? <a href="/signup">Sign Up</a></p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}
