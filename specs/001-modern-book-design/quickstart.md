# Quickstart Guide: Modern Book Design Implementation

## Overview
This guide provides step-by-step instructions to implement the modern book design for the Physical AI & Humanoid Robotics textbook website using Docusaurus.

## Prerequisites
- Node.js (v16 or higher)
- npm or yarn package manager
- Git
- Basic knowledge of React and CSS

## Setup and Installation

### 1. Clone and Navigate to Project
```bash
cd docusaurus-book
```

### 2. Install Dependencies (if not already installed)
```bash
npm install
```

## Implementation Steps

### Phase 1: Configuration Updates

#### 1. Update Docusaurus Configuration
1. Open `docusaurus.config.js`
2. Update the theme configuration with new color scheme:
   ```js
   themeConfig: {
     colorMode: {
       defaultMode: 'light',
       disableSwitch: false,
       respectPrefersColorScheme: true,
     },
     navbar: {
       title: 'Physical AI & Humanoid Robotics',
       logo: {
         alt: 'Physical AI & Humanoid Robotics Logo',
         src: 'img/logo.svg',
       },
       items: [
         // Update navigation items as needed
       ],
     },
     footer: {
       style: 'dark',
       links: [
         // Configure footer links
       ],
       copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook`,
     },
   }
   ```

#### 2. Update Color Variables
1. In `docusaurus.config.js`, add custom colors to themeConfig:
   ```js
   themeConfig: {
     // ... other config
     prism: {
       theme: require('prism-react-renderer/themes/github'),
       darkTheme: require('prism-react-renderer/themes/dracula'),
     },
   }
   ```

### Phase 2: Custom Styling

#### 1. Create Custom CSS File
1. Create `src/css/custom.css` if it doesn't exist
2. Add the following base styling:

```css
/**
 * Custom CSS for modern book design
 */

/* Color Variables */
:root {
  --ifm-color-primary: #2c5f9e;
  --ifm-color-primary-dark: #27558c;
  --ifm-color-primary-darker: #255084;
  --ifm-color-primary-darkest: #1e416b;
  --ifm-color-primary-light: #3d74ad;
  --ifm-color-primary-lighter: #477fb6;
  --ifm-color-primary-lightest: #6796c9;
  --ifm-color-secondary: #f0f8ff;
  --ifm-font-family-base: 'Georgia, serif';
  --ifm-font-family-monospace: 'SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace';
}

/* Typography for book-like appearance */
.docusaurus-markdown-content {
  font-family: 'Georgia, serif';
  line-height: 1.6;
  font-size: 1.1rem;
}

/* Homepage styling */
.hero {
  background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
  padding: 4rem 0;
  text-align: center;
}

.hero__title {
  font-family: 'Georgia, serif';
  font-weight: bold;
  color: #2c5f9e;
  font-size: 3rem;
}

.hero__subtitle {
  font-family: 'Georgia, serif';
  color: #555;
  font-size: 1.5rem;
  margin-top: 1rem;
}

/* Book-like content styling */
.markdown > h1 {
  font-family: 'Georgia, serif';
  border-bottom: 2px solid var(--ifm-color-primary);
  padding-bottom: 0.5rem;
}

.markdown > h2 {
  font-family: 'Georgia, serif';
  color: var(--ifm-color-primary);
}

.markdown > h3 {
  font-family: 'Georgia, serif';
  color: #444;
}

/* Navigation styling */
.navbar {
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Footer styling */
.footer {
  background-color: #2c5f9e;
  color: white;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .hero__title {
    font-size: 2rem;
  }

  .hero__subtitle {
    font-size: 1.2rem;
  }

  .docusaurus-markdown-content {
    font-size: 1rem;
  }
}
```

### Phase 3: Homepage Customization

#### 1. Create Custom Homepage
1. Create `src/pages/index.js` with the following content:

```jsx
import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Read the Textbook - 5min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics Textbook">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
```

### Phase 4: Component Customization

#### 1. Create Custom Components Directory
1. Create directory `src/components/`
2. Add custom components as needed for book-like features

#### 2. Create Homepage Features Component
1. Create `src/components/HomepageFeatures.js`:

```jsx
import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Physical AI',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Learn about AI Systems in the Physical World and Embodied Intelligence.
      </>
    ),
  },
  {
    title: 'Humanoid Robotics',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Explore the bridge between digital brain and physical body in robotics.
      </>
    ),
  },
  {
    title: 'Hands-On Learning',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Apply AI knowledge to control Humanoid Robots in simulated and real-world environments.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
```

### Phase 5: Add Assets

#### 1. Add Logo and Images
1. Place logo file in `static/img/logo.svg`
2. Add other images to `static/img/` as needed

### Phase 6: Testing

#### 1. Start Development Server
```bash
npm run start
```

#### 2. Build for Production
```bash
npm run build
```

## Verification Steps

1. Check homepage appearance - should have modern, book-like design
2. Verify navigation works properly
3. Test responsive design on mobile and tablet
4. Confirm all links function correctly
5. Validate color scheme and typography
6. Check that all images load properly
7. Verify accessibility features work

## Common Issues and Solutions

### Issue: Custom CSS not loading
**Solution**: Ensure `src/css/custom.css` is imported in the main layout or configuration

### Issue: Colors not applying
**Solution**: Check that CSS variables in `docusaurus.config.js` are properly formatted

### Issue: Responsive design problems
**Solution**: Verify media queries in custom CSS and test on multiple screen sizes

## Next Steps

1. Review and refine the design based on feedback
2. Add additional custom components as needed
3. Optimize images and assets for performance
4. Test accessibility features
5. Deploy to GitHub Pages