import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import SignupForm from '../components/Auth/SignupForm';

export default function Signup() {
  return (
    <Layout title="Signup" description="Create your account">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--6 col--offset-3">
            <div className="card shadow--md">
              <div className="card__header">
                <h2>Create Account</h2>
                <p>Join the Physical AI & Humanoid Robotics Course</p>
              </div>
              <div className="card__body">
                <SignupForm />
              </div>
              <div className="card__footer text--center">
                 <p>Already have an account? <Link to="/signin">Sign In</Link></p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}