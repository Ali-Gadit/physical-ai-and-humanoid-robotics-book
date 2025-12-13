import React, { useState } from 'react';
import { useHistory } from '@docusaurus/router';
import { authClient } from '../../lib/auth-client';

export default function SigninForm() {
  const history = useHistory();
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      await authClient.signIn.email({
        email: formData.email,
        password: formData.password
      }, {
        onSuccess: () => {
             history.push('/'); 
        },
        onError: (ctx) => {
            setError(ctx.error.message);
            setLoading(false);
        }
      });
    } catch (err: any) {
      setError(err.message || 'Signin failed');
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {error && <div className="alert alert--danger margin-bottom--sm">{error}</div>}
      
      <div className="margin-bottom--md">
        <label style={{display: 'block', marginBottom: '4px'}}>Email</label>
        <input 
          className="button button--block button--outline" 
          style={{textAlign: 'left', cursor: 'text', backgroundColor: 'var(--ifm-background-color)'}} 
          type="email" 
          name="email" 
          required 
          value={formData.email} 
          onChange={handleChange} 
        />
      </div>

       <div className="margin-bottom--md">
        <label style={{display: 'block', marginBottom: '4px'}}>Password</label>
        <input 
          className="button button--block button--outline" 
          style={{textAlign: 'left', cursor: 'text', backgroundColor: 'var(--ifm-background-color)'}} 
          type="password" 
          name="password" 
          required 
          value={formData.password} 
          onChange={handleChange} 
        />
      </div>

      <button type="submit" className="button button--primary button--block" disabled={loading}>
        {loading ? 'Signing in...' : 'Sign In'}
      </button>
    </form>
  );
}
