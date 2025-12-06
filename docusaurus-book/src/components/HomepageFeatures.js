import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Physical AI',
    description: (
      <>
        Explore AI Systems in the Physical World and Embodied Intelligence. Learn how to bridge the gap between digital brain and physical body.
      </>
    ),
    link: '/module1-ros2/overview',
  },
  {
    title: 'Humanoid Robotics',
    description: (
      <>
        Discover the fascinating world of Humanoid Robots and how AI knowledge can be applied to control them in simulated and real-world environments.
      </>
    ),
    link: '/module1-ros2/overview',
  },
  {
    title: 'Hands-On Learning',
    description: (
      <>
        Apply your knowledge through practical exercises and projects that connect theory with real implementation in robotics.
      </>
    ),
    link: '/module1-ros2/overview',
  },
];

function Feature({Svg, title, description, link}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
        {link && (
          <Link className="button button--primary button--outline" to={link}>
            Learn More
          </Link>
        )}
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