import React, { useState } from 'react';
import { useHistory } from '@docusaurus/router';
import { authClient } from '../../lib/auth-client';

export default function SignupForm() {
  const history = useHistory();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    preferredOs: 'Linux',
    hardwareEnvironment: 'Local',
    knowsPython: false,
    softwareSkillLevel: 'Beginner'
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    if (type === 'checkbox') {
        const checked = (e.target as HTMLInputElement).checked;
        setFormData(prev => ({ ...prev, [name]: checked }));
    } else {
        setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      await authClient.signUp.email({
        email: formData.email,
        password: formData.password,
        name: formData.name,
        // @ts-ignore
        softwareSkillLevel: formData.knowsPython ? formData.softwareSkillLevel : 'Beginner',
        preferredOs: formData.preferredOs,
        hardwareEnvironment: formData.hardwareEnvironment
      }, {
        onSuccess: () => {
             history.push('/physical-ai-and-humanoid-robotics-book/intro-physical-ai'); 
        },
        onError: (ctx) => {
            setError(ctx.error.message);
            setLoading(false);
        }
      });
    } catch (err: any) {
      setError(err.message || 'Signup failed');
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {error && <div className="alert alert--danger margin-bottom--sm">{error}</div>}
      
      <div className="margin-bottom--md">
        <label style={{display: 'block', marginBottom: '4px'}}>Name</label>
        <input 
          className="button button--block button--outline" 
          style={{textAlign: 'left', cursor: 'text', backgroundColor: 'var(--ifm-background-color)'}} 
          type="text" 
          name="name" 
          required 
          value={formData.name} 
          onChange={handleChange} 
        />
      </div>

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
      
      <div className="margin-bottom--md">
        <label style={{display: 'flex', alignItems: 'center', cursor: 'pointer'}}>
            <input 
                type="checkbox" 
                name="knowsPython" 
                checked={formData.knowsPython} 
                onChange={handleChange}
                style={{marginRight: '8px'}}
            />
            I have Python experience
        </label>
      </div>

      {formData.knowsPython && (
          <div className="margin-bottom--md">
            <label style={{display: 'block', marginBottom: '4px'}}>Python Skill Level</label>
            <select 
              className="button button--block button--outline" 
              style={{textAlign: 'left', backgroundColor: 'var(--ifm-background-color)'}}
              name="softwareSkillLevel" 
              value={formData.softwareSkillLevel} 
              onChange={handleChange}
            >
                <option value="Beginner">Beginner</option>
                <option value="Intermediate">Intermediate</option>
                <option value="Expert">Expert</option>
            </select>
          </div>
      )}

       <div className="margin-bottom--md">
        <label style={{display: 'block', marginBottom: '4px'}}>Preferred OS</label>
        <select 
          className="button button--block button--outline" 
          style={{textAlign: 'left', backgroundColor: 'var(--ifm-background-color)'}}
          name="preferredOs" 
          value={formData.preferredOs} 
          onChange={handleChange}
        >
            <option value="Linux">Linux</option>
            <option value="Windows">Windows</option>
            <option value="macOS">macOS</option>
            <option value="Other">Other</option>
        </select>
      </div>

       <div className="margin-bottom--md">
        <label style={{display: 'block', marginBottom: '4px'}}>Hardware Environment</label>
        <select 
          className="button button--block button--outline" 
          style={{textAlign: 'left', backgroundColor: 'var(--ifm-background-color)'}}
          name="hardwareEnvironment" 
          value={formData.hardwareEnvironment} 
          onChange={handleChange}
        >
            <option value="Local">Local Machine</option>
            <option value="Cloud">Cloud Environment</option>
        </select>
      </div>

      <button type="submit" className="button button--primary button--block" disabled={loading}>
        {loading ? 'Signing up...' : 'Sign Up'}
      </button>
    </form>
  );
}