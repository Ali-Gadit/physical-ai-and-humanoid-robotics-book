/**
 * Creating a sidebar enables you to:
 - Create an ordered group of docs
 - Render a sidebar from the docs folder structure
 - Include docs from different sources

- E.g. tour guides, api docs, etc.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Modules',
      items: [
        'module1-ros2',
        'module2-digital-twin',
        'module3-nvidia-isaac',
        'module4-vla',
      ],
    },
    {
      type: 'category',
      label: 'Weekly Breakdowns',
      items: [
        'week1-2-intro',
        'week3-5-ros2',
        'week6-7-gazebo',
        'week8-10-isaac',
        'week11-12-humanoid',
        'week13-conversational-robotics',
      ],
    },
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document', 'tutorial-basics/create-a-blog-post', 'tutorial-basics/create-a-page', 'tutorial-basics/deploy-your-site'],
    },
  ],
   */
};

module.exports = sidebars;
